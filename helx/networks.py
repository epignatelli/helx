from __future__ import annotations

from functools import wraps
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array

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
    representation_net: nn.Module = Identity()
    actor_net: nn.Module | None = None
    critic_net: nn.Module | None = None
    state_transition_net: nn.Module | None = None
    reward_net: nn.Module | None = None
    extra_net: nn.Module | None = None

    @nn.compact
    def __call__(
        self, observation: Array, *args, **kwargs
    ) -> Tuple[Array, Array, Array, Array]:
        modules = [
            self.actor_net,
            self.critic_net,
            self.state_transition_net,
            self.reward_net,
            self.extra_net,
        ]
        representation = self.representation_net(observation)
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
            self.apply(params, observation, method=self.representation_net, **kwargs)
        )

    def actor(self, params: nn.FrozenDict, observation: Array, **kwargs) -> Array:
        representation = self.apply(
            params, observation, method=self.representation_net, **kwargs
        )
        return jnp.asarray(
            self.apply(params, representation, method=self.actor_net, **kwargs)
        )

    def critic(self, params: nn.FrozenDict, observation: Array, **kwargs) -> Array:
        representation = self.apply(
            params, observation, method=self.representation_net, **kwargs
        )
        return jnp.asarray(
            self.apply(params, representation, method=self.critic_net, **kwargs)
        )

    def state_transition(
        self, params: nn.FrozenDict, observation: Array, action: Action, **kwargs
    ) -> Array:
        representation = self.apply(
            params, observation, method=self.representation_net, **kwargs
        )
        return jnp.asarray(
            self.apply(
                params,
                representation,
                action,
                method=self.state_transition_net,
                **kwargs,
            )
        )

    def reward(
        self, params: nn.FrozenDict, observation: Array, action: Action, **kwargs
    ) -> Array:
        representation = self.apply(
            params, observation, method=self.representation_net, **kwargs
        )
        return jnp.asarray(
            self.apply(params, representation, action, method=self.reward_net, **kwargs)
        )

    def extra(self, params: nn.FrozenDict, observation: Array, **kwargs) -> Array:
        representation = self.apply(
            params, observation, method=self.representation_net, **kwargs
        )
        return jnp.asarray(
            self.apply(params, representation, method=self.extra_net, **kwargs)
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
