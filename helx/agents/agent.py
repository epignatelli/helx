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
from os import PathLike
from typing import Any, Dict, Generic, Tuple, TypeVar

import flax.linen as nn
import jax
from chex import Array, Shape, dataclass
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


T = TypeVar("T", bound=Hparams)


class Agent(abc.ABC, Generic[T]):
    """Base class for a Reinforcement Learning agent. This class is meant to be subclassed
    for specific RL algorithms. It provides a common interface for training and evaluation
    of RL agents, and is compatible with the `helx` library for training and evaluation.
    Args:
        network (nn.Module): A Flax module that defines the network architecture.
        optimiser (GradientTransformation): An optax optimiser.
        hparams (Hparams): A dataclass that defines the hyperparameters of the agent.
        seed (int): A random seed for reproducibility.
    Properties:
        key (KeyArray): A JAX PRNG key.
        network (nn.Module): A Flax module that defines the network architecture.
        optimiser (GradientTransformation): An optax optimiser.
        hparams (Hparams): A dataclass that defines the hyperparameters of the agent.
        iteration (int): The current iteration of the agent.
        params (nn.FrozenDict): The current parameters of the agent.
        opt_state (OptState): The current optimiser state of the agent.
    """

    def __init__(
        self,
        hparams: T,
        network: AgentNetwork,
        optimiser: GradientTransformation,
        seed: int = 0,
    ):
        key: KeyArray = jax.random.PRNGKey(seed)
        obs = hparams.obs_space.sample(key)
        action = hparams.action_space.sample(key)
        outputs, params = network.init_with_output(key, obs, action, key)

        # properties:
        self.key: KeyArray = key
        self.network: AgentNetwork = network
        self.optimiser: GradientTransformation = optimiser
        self.hparams: T = hparams
        self.iteration: int = 0
        self.params: nn.FrozenDict = params  # type: ignore
        self.opt_state: OptState = optimiser.init(params)
        self.output_shape: Shape = jax.tree_map(lambda x: x.shape, list(outputs))  # type: ignore

        # methods:
        self.sgd_step = jax.jit(self._sgd_step)

    @abc.abstractmethod
    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        **kwargs,
    ) -> Tuple[Array, Any]:
        """The loss function to use for a minibatch gradient descent step.
        This function accepts a single _transition_ input, and not a minibatch,
        but it is automatically vectorised by `jax.vmap` when wrapped around `jax.value_and_grad`
        when calling `sgd_step`.
        Args:
            params (PyTreeDef): A pytree of function parameters.
            transition (Transition): A tuple of 6 elements (s, a, r, s', d, a')
                that defines MDP transitions from a state `s` through action aₜ,
                to a state `s'`, while collecting reward `r`. The term `d`
                indicates whether the state `s` is terminal,
                and aₜ₊₁ is an optional on-policy action.

        Returns:
            Returns a tuple of two elements containing, respectively:
            the calculated loss as a scalar, and any auxiliary variable to carry forward
            in the computation. Notice that once wrapped around `jax.value_and_grad`, the return
            type becomes a nested tuple `Tuple[Tuple[float, Any], PyTreeDef]`, with the last term
            being a `PyTree` of gradients with the same structure of `params`
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, episode: Trajectory) -> Dict[str, Any]:
        """Updates the agent state at the end of each episode and returns the loss as a scalar.
        This function can used to update both the agent's parameters and its memory.
        This function is usually not jittable, as we can not ensure that the agent memory, and other
        properties are jittable. This is also a good place to perform logging."""
        raise NotImplementedError()

    def name(self) -> str:
        """Returns the name of the agent."""
        return self.__class__.__name__

    def sample_action(self, env: Environment, eval: bool = False, **kwargs) -> Action:
        """Samples an action from the agent's policy or a set of actions if the environment is vectorised.
        Args:
            observation (Array): The observation to condition onto.
        Returns:
            Array: the action to take in the state s
        """
        observation = env.state()
        key = self._new_key(env.n_parallel())
        action, _ = self.network.actor(self.params, observation, key, **kwargs)
        return action

    def save(self, path: str | PathLike):
        # TODO(epignatelli): implement serialisation
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str | PathLike):
        # TODO(epignatelli): implement deserialisation
        raise NotImplementedError()

    def _sgd_step(
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
            # TODO(epignatelli): in_axis for named arguments
            # is not supported yet by jax.vmap
            # see https://github.com/google/jax/issues/7465
            in_axes = (None, 0) + (None,) * len(args)
            loss_fn = self.loss
            batch_loss = jax.vmap(loss_fn, in_axes=in_axes)
            loss, aux = batch_loss(params, transition, *args)
            return loss.mean(), aux

        backward = jax.value_and_grad(_loss, argnums=0, has_aux=True)
        (loss, aux), grads = backward(params, transition, *args)
        updates, opt_state = self.optimiser.update(grads, opt_state, params)
        params = apply_updates(params, updates)
        return params, opt_state, loss, aux

    def _new_key(self, n: int = 1):
        """Returns a new key from the PRNGKey."""
        self.key, k = jax.random.split(self.key)
        return jax.random.split(k, n).squeeze()
