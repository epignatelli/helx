"""Base classes for a Reinforcement Learning agent."""
from __future__ import annotations

import abc
from pydoc import locate
from typing import Any, Callable, Generic, NamedTuple, Tuple, TypeVar
import inspect

from flax.struct import PyTreeNode
import flax.linen as nn
import jax
from chex import Array, Shape
from jax.random import KeyArray
from optax import GradientTransformation, OptState
import optax
import rlax
import wandb

from helx.agents.agent import Hparams
from helx.memory import ReplayBuffer
from helx.networks.actors import EGreedyPolicy

from ..networks.modules import Identity

from ..mdp import Action, Episode, Transition
from ..networks import AgentNetwork, apply_updates
from ..spaces import Discrete, Space
from .. import __version__
from ..logging import get_logger
import jax.numpy as jnp

logging = get_logger()


class IValueBased:
    def critic(self, observation: Array, action: Action | None = None) -> Array:
        """Returns the value of the given state or state-action pair.
        Args:
            observation (Array): The observation to condition onto.
            action (Array): The action to condition onto.
        Returns:
            Array: the value of the state-action pair
        """
        raise NotImplementedError()


class IPolicyBased:
    def actor(self, observation: Array) -> Action:
        """Returns the action to take in the given state.
        Args:
            observation (Array): The observation to condition onto.
        Returns:
            Array: the action to take in the state s
        """
        raise NotImplementedError()


class IActorCritic(IValueBased, IPolicyBased):
    ...


class IModelBased:
    def state_transition(self, observation: Array, action: Array, **kwargs) -> Array:
        """Returns the next state given the current state and action.
        Args:
            observation (Array): The observation to condition onto.
            action (Array): The action to condition onto.
        Returns:
            Array: the next state
        """
        raise NotImplementedError()

    def reward(self, observation: Array, action: Array, **kwargs) -> Tuple[Array, Any]:
        """Returns the reward of the given state and action.
        Args:
            observation (Array): The observation to condition onto.
            action (Array): The action to condition onto.
        Returns:
            Array: the reward of the state-action pair
        """
        raise NotImplementedError()


class State(PyTreeNode):
    """The dynamic state, or simply _state_ in the context of stateless computations,
    of a component includes all its mutable variables: variables that
    can change after object initialisation. For example, it is unlikely
    that the computation graph of
    a function approximator of an agent changes after initialisation.
    Instead, the parameters of the function approximator are likely
    to be updated after initialisation.

    You can either create a new state object or inherit from this class
    to add additional fields.
    """

    step: int
    key: KeyArray


class SGDState(State):
    """The state of a SGD optimiser."""

    params: nn.FrozenDict
    opt_state: OptState


class IStateless:
    def init(self, *args, **kwargs) -> State:
        """Initialises the dynamic state of the object, i.e., the mutable variables.
        Args:
            key (KeyArray): The key to initialise the state with.
            observation (Array): The observation to condition onto.
        Returns:
            State: the initial state of the agent
        """
        raise NotImplementedError()


class IDeep:
    @abc.abstractmethod
    def loss(
        self, params: nn.FrozenDict, transition: Transition, **kwargs
    ) -> Tuple[Array, Any]:
        """Returns the loss of the agent on the given episode."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sgd_step(self, batch: Tuple[Array, ...], state: SGDState, **kwargs) -> State:
        """Performs a single SGD step on the network parameters.
        Args:
            key (KeyArray): A JAX PRNG key.
            batch (Tuple[Array, ...]): A batch of data to train on.
        Returns:
            Tuple[Array, Any]: A tuple of the loss and any extra information.
        """
        raise NotImplementedError()


class IAgent:
    @abc.abstractmethod
    def sample_action(
        self, observation: Array, key: KeyArray, eval: bool = False, **kwargs
    ) -> Action:
        """Samples an action from the agent's policy.
        Args:
            observation (Array): The observation to condition onto.
        Returns:
            Array: the action to take in the state s
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, episode: Episode) -> Array:
        """Updates the agent state at the end of each episode and returns the loss as a scalar.
        This function can used to update both the agent's parameters and its memory.
        This function is usually not jittable, as we can not ensure that the agent memory, and other
        properties are jittable. This is also a good place to perform logging."""
        raise NotImplementedError()


class ISerialisable:
    @abc.abstractmethod
    def serialise(self) -> dict[str, Any]:
        """Returns a dictionary representation of the agent."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def deserialise(cls, d: dict[str, Any]) -> ISerialisable:
        """Returns an instance of the agent from a dictionary representation."""
        raise NotImplementedError()
