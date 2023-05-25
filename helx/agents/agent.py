# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Base classes for a Reinforcement Learning agent."""
from __future__ import annotations

import abc
import pickle
from os import PathLike
from typing import Any, Dict, Generic, Tuple, TypeVar

import flax.linen as nn
import jax
from chex import Array, dataclass
from jax.random import KeyArray
from optax import GradientTransformation, OptState

from ..environment.base import Environment
from ..mdp import Action, Trajectory, Transition
from ..networks import AgentNetwork, apply_updates
from ..spaces import Space


@dataclass
class Hparams:
    """A base dataclass to define the hyperparameters of an agent."""

    obs_space: Space
    action_space: Space

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}



@dataclass
class AgentState:
    """The agent's dynamic state containing data that changes during training.
    Inherit from this class to add extra fields.
    The child class must be a `chex.dataclass`."""


T1 = TypeVar("T1", bound=Hparams)
T2 = TypeVar("T2", bound=AgentState)


@dataclass
class Agent(abc.ABC, Generic[T1, T2]):
    hparams: T1
    optimiser: GradientTransformation

    @abc.abstractclassmethod
    def init(cls, *args, **kwargs) -> T2:
        """Initialises the agent's state.
        This is the state that is updated at every learning step, and does not include
        variables that do not change during learning, such as the network architecture,
        the optimiser, and the hyperparameters.

        Returns:
            An `AgentState` or a subclass of it representing the agent's state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, agent_state: T2, episode: Trajectory) -> Tuple[AgentState, Dict[str, Any]]:
        """Updates the agent's state based on a trajectory of experience data,
        or on other parts of the agent's state, such as the replay buffer.
        Args:
            agent_state (AgentState): The agent's current state.
            episode (Trajectory): A trajectory of experience data *just* collected.
        Returns (Tuple[AgentState, Dict[str, Any]]):
            A tuple of two elements containing, respectively:
            the updated learning state, and a dictionary of metrics to log."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_action(self, agent_state: T2, env: Environment, eval: bool = False) -> Action:
        """Samples an action from the agent's policy or a set of actions if the environment is vectorised.
        Args:
            agent_state (AgentState): The agent's current state.
            env (Environment): The environment to sample actions from.
            eval (bool): Whether to sample actions for evaluation or training.
        Returns (Action):
            the action to take in the state s, with shape (n_envs, *action_shape)
            """
        raise NotImplementedError()

    @abc.abstractmethod
    def loss(self, agent_state: T2, transition: Transition, *args) -> Tuple[Array, Any]:
        """Defines the loss of the agent on a single (unbatched) transition.
        Args:
            agent_state (AgentState): The agent's current state.
            transition (Transition): A tuple of 5 elements (s, a, r, s', d).

        Returns (Tuple[Array, Any]):
            A tuple of two elements containing, respectively:
            the loss, and a dictionary of auxiliary metrics to log."""
        raise NotImplementedError()

    def sgd_step(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        opt_state: OptState,
        *args,
    ) -> Tuple[nn.FrozenDict, OptState, Array, Any]:
        """Performs a single SGD step on the agent's parameters.
        Args:
            params (PyTreeDef): A pytree of function parameters.
            transition (Transition): A *batch* of tuples of 5 elements (s, a, r, s', d)
                where the first axis is the batch axis.
            with an additional axis of size `batch_size` in position 0.
            opt_state (OptState): The optimiser state.
        Returns:
            Returns a tuple of two elements containing, respectively:
            the updated parameters as a pytree of the same structure as params,
            and the updated optimiser state.
        """
        def _loss(params, transition, *args):
            # see https://github.com/google/jax/issues/7465
            in_axes = (None, 0) + (None,) * len(args)
            batch_loss = jax.vmap(self.loss, in_axes=in_axes, axis_name="batch")
            loss, aux = batch_loss(params, transition, *args)
            return loss.mean(), aux

        backward = jax.value_and_grad(_loss, argnums=0, has_aux=True)
        (loss, aux), grads = backward(params, transition, *args)
        updates, opt_state = self.optimiser.update(grads, opt_state, params)
        params = apply_updates(params, updates)
        return params, opt_state, loss, aux

    def config(self):
        """Returns a dictionary of the dataclass properties defined by the agent"""
        return {k: v for k, v in self.__dataclass_fields__.items()}

    def name(self):
        """A printable version of the name of the agent."""
        return self.__class__.__name__

    def serialise(self, state: AgentState, path: str | PathLike) -> bytes:
        """Serialises the agent's config and state to a file using
            the latest pickle protocol."""
        agent = {
            "type": self.name(),
            "config": self.config(),
            "state": state}
        return pickle.dumps(agent)

    @classmethod
    def deserialise(cls, agent_bytes: bytes):
        """Loads the agent's config and state from pickled bytes.
        Returns a tuple of the agent's config and state."""
        agent_dict = pickle.loads(agent_bytes)
        agent = cls(**agent_dict["config"])
        state = agent_dict["state"]
        return agent, state



# T = TypeVar("T", bound=Hparams)



# class Agent(abc.ABC, Generic[T]):
#     """Base class for a Reinforcement Learning agent. This class is meant to be subclassed
#     for specific RL algorithms. It provides a common interface for training and evaluation
#     of RL agents, and is compatible with the `helx` library for training and evaluation.
#     Args:
#         network (nn.Module): A Flax module that defines the network architecture.
#         optimiser (GradientTransformation): An optax optimiser.
#         hparams (Hparams): A dataclass that defines the hyperparameters of the agent.
#         seed (int): A random seed for reproducibility.
#     Properties:
#         key (KeyArray): A JAX PRNG key.
#         network (nn.Module): A Flax module that defines the network architecture.
#         optimiser (GradientTransformation): An optax optimiser.
#         hparams (Hparams): A dataclass that defines the hyperparameters of the agent.
#         iteration (int): The current iteration of the agent.
#         params (nn.FrozenDict): The current parameters of the agent.
#         opt_state (OptState): The current optimiser state of the agent.
#     """

#     def __init__(
#         self,
#         hparams: T,
#         network: nn.Module,
#         optimiser: GradientTransformation,
#         seed: int = 0,
#     ):
#         key: KeyArray = jax.random.PRNGKey(seed)
#         obs = hparams.obs_space.sample(key)
#         action = hparams.action_space.sample(key)
#         outputs, params = network.init_with_output(key, obs, action, key)

#         # properties:
#         self.key: KeyArray = key
#         self.network: nn.Module = network
#         self.optimiser: GradientTransformation = optimiser
#         self.hparams: T = hparams
#         self.iteration: int = 0
#         self.params: nn.FrozenDict = params  # type: ignore
#         self.opt_state: OptState = optimiser.init(params)
#         self.output_shape: Shape = jax.tree_map(lambda x: x.shape, list(outputs))  # type: ignore

#         # methods:
#         self.sgd_step = jax.jit(self._sgd_step)

#     @abc.abstractmethod
#     def loss(
#         self,
#         params: nn.FrozenDict,
#         transition: Transition,
#         **kwargs,
#     ) -> Tuple[Array, Any]:
#         """The loss function to use for a minibatch gradient descent step.
#         This function accepts a single _transition_ input, and not a minibatch,
#         but it is automatically vectorised by `jax.vmap` when wrapped around `jax.value_and_grad`
#         when calling `sgd_step`.

#         Args:
#             params (PyTreeDef): A pytree of function parameters.
#             transition (Transition): A tuple of 6 elements (s, a, r, s', d, a')
#                 that defines MDP transitions from a state `s` through action aₜ,
#                 to a state `s'`, while collecting reward `r`. The term `d`
#                 indicates whether the state `s` is terminal,
#                 and aₜ₊₁ is an optional on-policy action.

#         Returns:
#             Returns a tuple of two elements containing, respectively:
#             the calculated loss as a scalar, and any auxiliary variable to carry forward
#             in the computation. Notice that once wrapped around `jax.value_and_grad`, the return
#             type becomes a nested tuple `Tuple[Tuple[float, Any], PyTreeDef]`, with the last term
#             being a `PyTree` of gradients with the same structure of `params`
#         """
#         raise NotImplementedError()

#     @abc.abstractmethod
#     def update(self, state: Any, episode: Trajectory) -> Tuple[Any, Dict[str, Any]]:
#         """Updates the agent state at the end of each episode, given the current
#         learning state `state` and some experience data `episode`.

#         Args:
#             state (Any): The current learning state of the agent.
#                 This usually includes the network parameters, the optimiser state,
#                 and can also include a replay buffer, a counter, or extra parameters
#                 for the learning algorithm.
#                 It does not contain variables that do not change during learning,
#                 such as the network architecture, the optimiser, and the hyperparameters.
#             episode (Trajectory): A trajectory of experience data *just* collected.

#         Returns:
#             A tuple of two elements containing, respectively:
#             the updated learning state, and a dictionary of metrics to log.
#         """
#         raise NotImplementedError()

#     @abc.abstractmethod
#     def state(self) -> Any:
#         """Returns the agent's current state.
#         This is the state that is updated at every learning step, and does not include
#         variables that do not change during learning, such as the network architecture,
#         the optimiser, and the hyperparameters.

#         Returns:
#             A pytree representing the agent's state."""
#         raise NotImplementedError()

#     def name(self) -> str:
#         """Gets the name of the agent."""
#         return self.__class__.__name__

#     def sample_action(self, env: Environment, eval: bool = False, **kwargs) -> Action:
#         """Samples an action from the agent's policy or a set of actions if the environment is vectorised.
#         Args:
#             observation (Array): The observation to condition onto.
#         Returns:
#             Array: the action to take in the state s
#         """
#         observation = env.state()
#         key = self._new_key(env.n_parallel())
#         action, _ = self.network.actor(self.params, observation, key, **kwargs)
#         return action

#     def save(self, path: str | PathLike):
#         # TODO(epignatelli): implement serialisation
#         raise NotImplementedError()

#     @classmethod
#     def load(cls, path: str | PathLike):
#         # TODO(epignatelli): implement deserialisation
#         raise NotImplementedError()

#     def _sgd_step(
#         self,
#         params: nn.FrozenDict,
#         transition: Transition,
#         opt_state: OptState,
#         *args,
#     ) -> Tuple[nn.FrozenDict, OptState, Array, Any]:
#         """Performs a single SGD step on the agent's parameters.
#         Args:
#             params (PyTreeDef): A pytree of function parameters.
#             transition (Transition): A *batch* of tuples of 5 elements (s, a, r, s', d)
#                 where the first axis is the batch axis.
#             with an additional axis of size `batch_size` in position 0.
#             opt_state (OptState): The optimiser state.
#         Returns:
#             Returns a tuple of two elements containing, respectively:
#             the updated parameters as a pytree of the same structure as params,
#             and the updated optimiser state.
#         """
#         def _loss(params, transition, *args):
#             # TODO(epignatelli): in_axis for named arguments
#             # is not supported yet by jax.vmap
#             # see https://github.com/google/jax/issues/7465
#             in_axes = (None, 0) + (None,) * len(args)
#             loss_fn = self.loss
#             batch_loss = jax.vmap(loss_fn, in_axes=in_axes)
#             loss, aux = batch_loss(params, transition, *args)
#             return loss.mean(), aux

#         backward = jax.value_and_grad(_loss, argnums=0, has_aux=True)
#         (loss, aux), grads = backward(params, transition, *args)
#         updates, opt_state = self.optimiser.update(grads, opt_state, params)
#         params = apply_updates(params, updates)
#         return params, opt_state, loss, aux

#     def _new_key(self, n: int = 1):
#         """Returns a new key from the PRNGKey."""
#         self.key, k = jax.random.split(self.key)
#         return jax.random.split(k, n).squeeze()
