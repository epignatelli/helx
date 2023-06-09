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
from typing import Any, Dict, Generic, List, Tuple, TypeVar

import flax.linen as nn
from flax import struct
import jax
from chex import Array, dataclass
from jax.random import KeyArray
from optax import GradientTransformation, OptState

from ..environment.base import Environment
from ..mdp import Action, Timestep, Transition
from ..networks import AgentNetwork, apply_updates
from ..spaces import Space


class IAgent(abc.ABC, struct.PyTreeNode):
    """The Agent's interface.
    To implement your own agent, you can either implement this interface,
    or subclass the `Agent` class, which provides additional useful methods."""

    hparams: Hparams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)

    @abc.abstractclassmethod
    def create(cls, *args, **kwargs) -> Agent:
        """Use this method to construct an agent."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_action(
        self, env: Environment, key: KeyArray, eval: bool = False
    ) -> Action:
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
    def update(
        self, rollouts: List[Timestep], key: KeyArray
    ) -> Tuple[Agent, Dict[str, Any]]:
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
    def loss(self, transition: Transition, *args) -> Tuple[Array, Any]:
        """Defines the loss of the agent on a single (unbatched) transition.
        Args:
            agent_state (AgentState): The agent's current state.
            transition (Transition): A tuple of 5 elements (s, a, r, s', d).

        Returns (Tuple[Array, Any]):
            A tuple of two elements containing, respectively:
            the loss, and a dictionary of auxiliary metrics to log."""
        raise NotImplementedError()


class Hparams(struct.PyTreeNode):
    """A base dataclass to define the hyperparameters of an agent."""

    obs_space: Space
    action_space: Space
    discount: float = 1.0
    n_steps: int = 1
    seed: int = 0

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class Agent(IAgent):
    def sgd_step(
        self,
        params: Any,
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

    def name(self):
        """A printable version of the name of the agent."""
        return self.__class__.__name__

    def serialise(self, path: str | PathLike) -> bytes:
        """Serialises the agent's config and state to a file using
        the latest pickle protocol."""
        return pickle.dumps(self)

    @classmethod
    def deserialise(cls, agent_bytes: bytes) -> Agent:
        """Loads the agent's config and state from pickled bytes.
        Returns a tuple of the agent's config and state."""
        agent_dict = pickle.loads(agent_bytes)
        agent = cls(**agent_dict)
        return agent

    def save(self, path: str | PathLike):
        with open(path, "wb") as f:
            f.write(self.serialise(path))
        return

    @classmethod
    def load(cls, path: str | PathLike):
        with open(path, "rb") as f:
            agent_bytes = f.read()
        return cls.deserialise(agent_bytes)
