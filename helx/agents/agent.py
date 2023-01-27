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
from typing import Any, Generic, Tuple, TypeVar

import flax.linen as nn
import jax
from chex import Array, Shape, dataclass
from jax.random import KeyArray
from optax import GradientTransformation, OptState

from ..mdp import Action, Episode, Transition
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
        self.params: nn.FrozenDict = params
        self.opt_state: OptState = optimiser.init(params)
        self.output_shape: Shape = jax.tree_map(lambda x: x.shape, list(outputs))

        # methods:
        self.sgd_step = jax.jit(self._sgd_step)

    @abc.abstractmethod
    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        **kwargs,
    ) -> Tuple[Array, Any]:
        """The loss function to differentiate through for Deep RL agents.
            This function can be used for heavy computation and can be xla-compiled,
            and is always *jitted by default*.
            A possible design pattern is with decorators:
        Example:
        ```
            >>> @partial(jax.value_and_grad, argnums=1, has_aux=True)
            >>> def loss(self, params, sarsa, *, **kwargs):
            >>>     s, a, r_tp1, s_tp1, d_tp1, a_tp1 = sarsa
            >>>     return rlax.l2_loss(self.network.apply(params, s)).mean()
        ```
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
    def update(self, episode: Episode) -> Array:
        """Updates the agent state at the end of each episode and returns the loss as a scalar.
        This function can used to update both the agent's parameters and its memory.
        This function is usually not jittable, as we can not ensure that the agent memory, and other
        properties are jittable. This is also a good place to perform logging."""
        raise NotImplementedError()

    def sample_action(self, observation: Array, eval: bool = False, **kwargs) -> Action:
        """Samples an action from the agent's policy.
        Args:
            observation (Array): The observation to condition onto.
        Returns:
            Array: the action to take in the state s
        """
        action, _ = self.network.actor(
            self.params, observation, self._new_key(), **kwargs
        )
        return action

    def save(self, path):
        # TODO(epignatelli): implement
        raise NotImplementedError()

    @classmethod
    def load(cls, path):
        # TODO(epignatelli): implement
        raise NotImplementedError()

    def _loss_batched(
        self,
        params: nn.FrozenDict,
        batched_transitions: Transition,
        *args,
    ) -> Tuple[Array, Any]:
        """A batched version of the loss function.
        The returned `loss` value is reduced with a `jnp.mean` operation,
        while the returned `aux` value is returned as it is.

        Args:
            params (PyTreeDef): A Flax pytree of module parameters
            batched_transitions (Transition): A tuple of 5 elements (s, a, r, s', d)
            with an additional axis of size `batch_size` in position 0.

        Returns (Tuple[Array, PyTreeDef]):
            Returns a tuple of two elements containing, respectively:
            the average loss across the minibatch, and a PyTreeDef containing the aux
            variables returned."""
        # in_axis for named arguments is not supported yet by jax.vmap
        # see https://github.com/google/jax/issues/7465
        in_axes = (None, 0) + (None,) * len(args)
        batch_loss = jax.vmap(self.loss, in_axes=in_axes)
        loss, aux = batch_loss(params, batched_transitions, *args)
        return loss.mean(), aux

    def _sgd_step(
        self,
        params: nn.FrozenDict,
        batched_transition: Transition,
        opt_state: OptState,
        *args,
    ) -> Tuple[nn.FrozenDict, OptState, Array, Any]:
        """Performs a single SGD step on the agent's parameters.
        Args:
            params (PyTreeDef): A pytree of function parameters.
            batched_transition (Transition): A tuple of 5 elements (s, a, r, s', d)
            with an additional axis of size `batch_size` in position 0.
            opt_state (OptState): The optimiser state.
        Returns:
            Returns a tuple of two elements containing, respectively:
            the updated parameters as a pytree of the same structure as params,
            and the updated optimiser state.
        """
        backward = jax.value_and_grad(self._loss_batched, argnums=0, has_aux=True)
        (loss, aux), grads = backward(params, batched_transition, *args)
        updates, opt_state = self.optimiser.update(grads, opt_state, params)
        params = apply_updates(params, updates)
        return params, opt_state, loss, aux

    def _new_key(self, n: int = 1):
        """Returns a new key from the PRNGKey."""
        self.key, k = jax.random.split(self.key)
        return jax.random.split(k, n).squeeze()
