"""Base class for a Reinforcement Learning agent."""
from __future__ import annotations

import abc
from functools import wraps
from typing import Any, Generic, Tuple, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, Shape
from jax.random import KeyArray
from optax import GradientTransformation, OptState

from ..mdp import Episode, Transition


@wraps(optax.apply_updates)
def _apply_updates(params: nn.FrozenDict, updates: optax.Updates) -> nn.FrozenDict:
    # work around a typing compatibility issue
    # optax.apply_updates expects a pytree of parameters
    # but works with nn.FrozenDict anyway
    return optax.apply_updates(params, updates)  # type: ignore


class Hparams:
    input_shape: Shape

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
        network: nn.Module,
        optimiser: GradientTransformation,
        hparams: T,
        seed: int,
    ):
        # init:
        key: KeyArray = jax.random.PRNGKey(seed)
        params: nn.FrozenDict = network.init(
            key, jnp.ones(hparams.input_shape, dtype=jnp.float32)
        )

        # const:
        self.key: KeyArray = key
        self.network: nn.Module = network
        self.optimiser: GradientTransformation = optimiser
        self.hparams: T = hparams
        self.iteration: int = 0
        self.params: nn.FrozenDict = params
        self.opt_state: OptState = optimiser.init(params)

        self.sgd_step = jax.jit(self._sgd_step)

    @abc.abstractmethod
    def policy(
        self, params: nn.FrozenDict, observation: Array, eval=False, **kwargs
    ) -> Array:
        """The policy function to evaluate the agent's policy π(s).
        Args:
            params (PyTreeDef): A pytree of function parameters.
            s (Array): The state to evaluate the policy on.
        Returns:
            Array: the action to take in the state s
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_action(self, observation: Array, eval: bool = False, **kwargs) -> Array:
        """Collects a sample from the policy distribution conditioned on the provided observation π(s), and
            returns both the sampled action, and a (possibly empty) array of the correspondent log probabilities.
        Args:
            observation (Array): The observation to condition onto.
            keys (PRNGKey): a random key used to sample the action
            eval (bool): optional boolean if the agent follows a different policy at evaluation time
        Returns:
            Array: the sampled action
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, episode: Episode) -> Array:
        """Updates the agent state at the end of each episode and returns the loss as a scalar.
        This function can used to update both the agent's parameters and its memory.
        This function is usually not jittable, as we can not ensure that the agent memory, and other
        properties are jittable. This is also a good place to perform logging."""
        raise NotImplementedError()

    @abc.abstractmethod
    def loss(
        self, params: nn.FrozenDict, transition: Transition, **kwargs
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

    def _loss(
        self,
        params: nn.FrozenDict,
        batched_transitions: Transition,
        *args,
    ) -> Tuple[Array, Any]:
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
            batched_transition (Tuple): A tuple of 5 elements (s, a, r, s', d)
            with an additional axis of size `batch_size` in position 0.
            opt_state (OptState): The optimiser state.
        Returns:
            Returns a tuple of two elements containing, respectively:
            the updated parameters as a pytree of the same structure as params,
            and the updated optimiser state.
        """
        backward = jax.value_and_grad(self._loss, argnums=0, has_aux=True)
        (loss, aux), grads = backward(params, batched_transition, *args)
        updates, opt_state = self.optimiser.update(grads, opt_state, params)
        params = _apply_updates(params, updates)
        return params, opt_state, loss, aux

    def new_key(self, n: int = 1):
        self.key, k = jax.random.split(self.key)
        return jax.random.split(k, n).squeeze()

    def save(self, path):
        raise NotImplementedError()

    @classmethod
    def load(cls, path):
        raise NotImplementedError()
