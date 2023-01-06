from __future__ import annotations

from functools import wraps
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, PyTreeDef

from .mdp import Action


@wraps(jax.jit)
def jit_bound(method, *args, **kwargs):
    return jax.jit(nn.module._get_unbound_fn(method), *args, **kwargs)


@wraps(optax.apply_updates)
def apply_updates(params: nn.FrozenDict, updates: optax.Updates) -> nn.FrozenDict:
    # work around a typing compatibility issue
    # optax.apply_updates expects a pytree of parameters
    # but works with nn.FrozenDict anyway
    return optax.apply_updates(params, updates)  # type: ignore


class Flatten(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x.flatten()


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x


class AgentNetwork(nn.Module):
    actor_net: nn.Module | None = None
    critic_net: nn.Module | None = None
    state_transition_net: nn.Module | None = None
    reward_net: nn.Module | None = None
    extra_net: nn.Module | None = None
    state_representation_net: nn.Module = Identity()

    @nn.compact
    def __call__(
        self, observation: Array, *args, **kwargs
    ) -> Tuple[Array, Array, Array, Array, PyTreeDef]:
        modules = [
            self.actor_net,
            self.critic_net,
            self.state_transition_net,
            self.reward_net,
            self.extra_net,
        ]
        representation = self.state_representation_net(observation)
        ys = []
        for module in modules:
            if module is None:
                y = jnp.empty((0,))
            else:
                y = module(representation, *args, **kwargs)
            ys.append(y)
        return tuple(ys)

    def state_representation(
        self, params: nn.FrozenDict, observation: Array, **kwargs
    ) -> Array:
        return jnp.asarray(
            self.state_representation_net.apply(
                {"params": params["params"]["representation_net"]},
                observation,
                **kwargs,
            )
        )

    def actor(
        self,
        params: nn.FrozenDict,
        observation: Array,
        apply_representation=False,
        **kwargs,
    ) -> Array:
        if self.actor_net is None:
            raise ValueError("Actor net not defined")
        if apply_representation:
            observation = self.state_representation(params, observation, **kwargs)
        return jnp.asarray(
            self.actor_net.apply(
                {"params": params["params"]["actor_net"]},
                observation,
                **kwargs,
            )
        )

    def critic(
        self,
        params: nn.FrozenDict,
        observation: Array,
        apply_representation=False,
        **kwargs,
    ) -> Array:
        if self.critic_net is None:
            raise ValueError("Critic net not defined")
        if apply_representation:
            observation = self.state_representation(params, observation, **kwargs)
        return jnp.asarray(
            self.critic_net.apply(
                {"params": params["params"]["critic_net"]},
                observation,
                **kwargs,
            )
        )

    def state_transition(
        self,
        params: nn.FrozenDict,
        observation: Array,
        action: Action,
        apply_representation=False,
        **kwargs,
    ) -> Array:
        if self.state_transition_net is None:
            raise ValueError("State transition net not defined")
        if apply_representation:
            observation = self.state_representation(params, observation, **kwargs)
        return jnp.asarray(
            self.state_transition_net.apply(
                {"params": params["params"]["state_transition_net"]},
                observation,
                action,
                **kwargs,
            )
        )

    def reward(
        self,
        params: nn.FrozenDict,
        observation: Array,
        action: Action,
        apply_representation=False,
        **kwargs,
    ) -> Array:
        if self.reward_net is None:
            raise ValueError("Reward net not defined")
        if apply_representation:
            observation = self.state_representation(params, observation, **kwargs)
        return jnp.asarray(
            self.reward_net.apply(
                {"params": params["params"]["reward_net"]},
                observation,
                action,
                **kwargs,
            )
        )

    def extra(
        self,
        params: nn.FrozenDict,
        observation: Array,
        apply_representation=False,
        **kwargs,
    ) -> Array:
        if self.extra_net is None:
            raise ValueError("Extra net not defined")
        if apply_representation:
            observation = self.state_representation(params, observation, **kwargs)
        return jnp.asarray(
            self.extra_net.apply(
                {"params": params["params"]["extra_net"]},
                observation,
                **kwargs,
            )
        )


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i in range(len(self.features)):
            x = nn.Dense(features=self.features[i])(x)
            x = nn.relu(x)
        return x


class CNN(nn.Module):
    features: Tuple[int] = (32,)
    kernel_sizes: Tuple[Tuple[int, ...]] = ((3, 3),)
    strides: Tuple[Tuple[int, ...]] = ((1, 1),)
    padding: Tuple[nn.linear.PaddingLike] = ("SAME",)

    def setup(self):
        assert (
            len(self.features)
            == len(self.kernel_sizes)
            == len(self.strides)
            == len(self.padding)
        ), "All arguments must have the same length, got {} {} {} {}".format(
            len(self.features),
            len(self.kernel_sizes),
            len(self.strides),
            len(self.padding),
        )
        self.modules = [
            nn.Conv(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                strides=self.strides[i],
                padding=self.padding[i],
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
            x = nn.relu(x)
        return x


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> Array:
        log_temperature = self.param(
            "log_temperature",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temperature)
