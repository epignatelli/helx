from __future__ import annotations

from functools import wraps
from typing import Callable, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, PyTreeDef
from jax.random import KeyArray

from ..mdp import Action


@wraps(jax.jit)
def jit_bound(method, *args, **kwargs):
    """A wrapper for jax.jit that allows jitting bound methods, such as
    instance methods."""
    return jax.jit(nn.module._get_unbound_fn(method), *args, **kwargs)


@wraps(optax.apply_updates)
def apply_updates(params: nn.FrozenDict, updates: optax.Updates) -> nn.FrozenDict:
    """Applies updates to parameters using optax. This is a workaround for a typing
    compatibility issue between optax and Flax: optax.apply_updates expects a pytree
    of parameters, but works with nn.FrozenDict anyway.
    Args:
        params (nn.FrozenDict): A pytree of Flax parameters.
        updates (optax.Updates): A pytree of updates.
    Returns:
        nn.FrozenDict: The update pytree of Flax parameters.
    """
    return optax.apply_updates(params, updates)  # type: ignore


class Flatten(nn.Module):
    """A Flax module that flattens the input array."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x.flatten()


class Identity(nn.Module):
    """A Flax module that returns the input array."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x


class MLP(nn.Module):
    """Defines a multi-layer perceptron by alternating a dense layer with a ReLU
    non-linearity.
    Args:
        features (Sequence[int]): A sequence of integers that define the number of
            units in each dense layer. E.g., (32, 64) defines a network with two
            dense layers, the first with 32 units and the second with 64.
    Example:
        >>> import jax.numpy as jnp
        >>> from helx.networks import MLP
        >>> mlp = MLP(features=(32, 64))
        >>> x = jnp.ones((1, 28, 28))
        >>> mlp(x).shape
        (1, 64)"""

    features: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, x: Array, *args, **kwargs) -> Array:
        for i in range(len(self.features)):
            x = nn.Dense(features=self.features[i])(x)
            x = self.activation(x)
        return x


class CNN(nn.Module):
    """Defines a convolutional neural network by alternating a
    convolution with a ReLU non-linearity.
    There is no pooling or other non-linearities between convolutions.
    Args:
        features (Tuple[int, ...]): A sequence of integers that define the number of
            filters in each convolution. E.g., (32, 64) defines a network with
            two convolutions, the first with 32 filters and the second with 64.
        kernel_sizes (Tuple[Tuple[int, ...]]): A sequence of tuples that define
            the kernel size of each convolution.
        strides (Tuple[Tuple[int, ...]]): A sequence of tuples that define the
            stride of each convolution.
        padding (Tuple[nn.linear.PaddingLike]): A sequence of padding modes for
            each convolution.

    Example:
        >>> import jax.numpy as jnp
        >>> from helx.networks import CNN
        >>> cnn = CNN(
        ...     features=(32, 64),
        ...     kernel_sizes=((3, 3), (3, 3)),
        ...     strides=((1, 1), (1, 1)),
        ...     padding=("SAME", "SAME"),
        ... )
        >>> cnn(jnp.ones((1, 10, 10, 3))).shape
        (1, 10, 10, 64)
    """

    features: Tuple[int, ...] = (32,)
    kernel_sizes: Tuple[Tuple[int, ...], ...] = ((3, 3),)
    strides: Tuple[Tuple[int, ...], ...] = ((1, 1),)
    paddings: Tuple[nn.linear.PaddingLike, ...] = ("SAME",)
    activation: Callable[[Array], Array] = nn.relu
    flatten: bool = False

    def setup(self):
        assert (
            len(self.features)
            == len(self.kernel_sizes)
            == len(self.strides)
            == len(self.paddings)
        ), "All arguments must have the same length, got {} {} {} {}".format(
            len(self.features),
            len(self.kernel_sizes),
            len(self.strides),
            len(self.paddings),
        )
        self.modules = [
            nn.Conv(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                strides=self.strides[i],
                padding=self.paddings[i],
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        for module in self.modules:
            x = module(x)
            x = self.activation(x)
        if self.flatten:
            x = x.flatten()
        return x


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self, observation: Array, action: Array) -> Array:
        log_temperature = self.param(
            "log_temperature",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temperature)


class AgentNetwork(nn.Module):
    """Defines the network architecture of an agent, and can be used as it is.
    Args:
        actor_net (nn.Module): A Flax module that defines the actor network.
            The signature of this module should be `f(Array, KeyArray)`.
        critic_net (nn.Module): A Flax module that defines the critic network.
            The signature of this module should be `f(Array)`.
        state_transition_net (nn.Module): A Flax module that defines the
            state-transition dynamics network. Used for model-based RL.
            The signature of this module should be `f(Array, Action)`.
        reward_net (nn.Module): A Flax module that defines the reward network.
            Used for model-based RL.
            The signature of this module should be `f(Array, Action)`.
        extra_net (nn.Module): A Flax module that computes custom, unstructured data
            (e.g. a log of the agent's actions, additional rewards, goals).
            The signature of this module should be `f(Array, Action)`.
        state_representation_net (nn.Module): A Flax module that defines the state
            representation network. If the representation is shared between the actor
            and critic, then this module should be an instance of `Identity`, and
            the input of actor_net and critic_net should be the raw observation.
    """

    actor_net: nn.Module | None = None
    critic_net: nn.Module | None = None
    state_transition_net: nn.Module | None = None
    reward_net: nn.Module | None = None
    extra_net: nn.Module | None = None
    state_representation_net: nn.Module = Identity()

    @nn.compact
    def __call__(
        self, observation: Array, action: Action, key
    ) -> Tuple[Array, Array, Array, Array, Array | PyTreeDef, Array]:
        representation = self.state_representation_net(observation)

        actor_out = jnp.empty((0,))
        if self.actor_net is not None:
            actor_out = self.actor_net(representation, key)

        critic_out = jnp.empty((0,))
        if self.critic_net is not None:
            critic_out = self.critic_net(representation)

        state_transition_out = jnp.empty((0,))
        if self.state_transition_net is not None:
            state_transition_out = self.state_transition_net(representation, action)

        reward_out = jnp.empty((0,))
        if self.reward_net is not None:
            reward_out = self.reward_net(representation, action)

        extra_out = jnp.empty((0,))
        if self.extra_net is not None:
            extra_out = self.extra_net(observation, action)

        return (
            actor_out,
            critic_out,
            state_transition_out,
            reward_out,
            extra_out,
            representation,
        )

    def state_representation(self, params: nn.FrozenDict, observation: Array) -> Array:
        if "state_representation_net" not in params["params"]:
            return observation
        return jnp.asarray(
            self.state_representation_net.apply(
                {"params": params["params"]["state_representation_net"]},
                observation,
            )
        )

    def actor(
        self,
        params: nn.FrozenDict,
        observation: Array,
        key: KeyArray,
    ) -> Array | PyTreeDef:
        if self.actor_net is None:
            raise ValueError("Actor net not defined")
        observation = self.state_representation(params, observation)
        return jnp.asarray(
            self.actor_net.apply(
                {"params": params["params"]["actor_net"]}, observation, key
            )
        )

    def critic(
        self,
        params: nn.FrozenDict,
        observation: Array,
    ) -> Array | PyTreeDef:
        if self.critic_net is None:
            raise ValueError("Critic net not defined")
        observation = self.state_representation(params, observation)
        return jnp.asarray(
            self.critic_net.apply(
                {"params": params["params"]["critic_net"]},
                observation,
            )
        )

    def state_transition(
        self,
        params: nn.FrozenDict,
        observation: Array,
        action: Action,
    ) -> Array | PyTreeDef:
        if self.state_transition_net is None:
            raise ValueError("State transition net not defined")
        observation = self.state_representation(params, observation)
        return jnp.asarray(
            self.state_transition_net.apply(
                {"params": params["params"]["state_transition_net"]},
                observation,
                action,
            )
        )

    def reward(
        self,
        params: nn.FrozenDict,
        observation: Array,
        action: Action,
    ) -> Array:
        if self.reward_net is None:
            raise ValueError("Reward net not defined")
        observation = self.state_representation(params, observation)
        return jnp.asarray(
            self.reward_net.apply(
                {"params": params["params"]["reward_net"]},
                observation,
                action,
            )
        )

    def extra(
        self,
        params: nn.FrozenDict,
        observation: Array | None = None,
        action: Action | None = None,
    ) -> Array | PyTreeDef:
        if self.extra_net is None:
            raise ValueError("Extra net not defined")
        return jnp.asarray(
            self.extra_net.apply(
                {"params": params["params"]["extra_net"]},
                observation,
                action,
            )
        )
