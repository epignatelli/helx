"""Base class for a Reinforcement Learning agent."""
from __future__ import annotations

import abc
from typing import Any, Tuple

import jax
from chex import Array, dataclass
from flax.core.scope import FrozenVariableDict

from ..environment.mdp import Episode


class Params(FrozenVariableDict):
    ...


class Hparams:
    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class Agent(abc.ABC):
    iteration: int
    hparams: Hparams

    @abc.abstractmethod
    def loss(
        self, params: Params, sarsa: Episode, *args, **kwargs
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
            sarsa (Tuple): A tuple of 6 elements (sₜ, aₜ, rₜ₊₁, sₜ₊₁, dₜ₊₁, aₜ₊₁)
                that defines MDP transitions from a state sₜ through action aₜ,
                to a state sₜ₊₁, while collecting reward rₜ₊₁. The term dₜ₊₁
                indicates whether the state sₜ₊₁ is terminal,
                and aₜ₊₁ is an optional on-policy action.

        Returns:
            Returns a tuple of two elements containing, respectively:
            the calculated loss as a scalar, and any auxiliary variable to carry forward
            in the computation. Notice that once wrapped around `jax.value_and_grad`, the return
            type becomes a nested tuple `Tuple[Tuple[float, Any], PyTreeDef]`, with the last term
            being a `PyTree` of gradients with the same structure of `params`
        """

    @abc.abstractmethod
    def sample_action(
        self, observation: Array, eval: bool = False, **kwargs
    ) -> Tuple[Array, Array]:
        """Collects a sample from the policy distribution conditioned on the provided observation π(s), and
            returns both the sampled action, and a (possibly empty) array of the correspondent log probabilities.
        Args:
            observation (Array): The observation to condition onto.
            keys (PRNGKey): a random key used to sample the action
            eval (bool): optional boolean if the agent follows a different policy at evaluation time
        Returns:
            Tuple[Array, Array]: a tuple containing
                i) the sampled action and
                ii) the log probability of the sampled action, for one-action-out architectures
                    the log probabilities for all actions for all-actions-out architectures"""

    @abc.abstractmethod
    def update(self, episode: Episode) -> Array:
        """Updates the agent state at the end of each episode and returns the loss as a scalar.
        This function can used to update both the agent's parameters and its memory.
        This function is usually not jittable, as we can not ensure that the agent memory, and other
        properties are jittable. This is also a good place to perform logging."""

    def new_key(self, n: int = 1):
        self.key, k = jax.random.split(self.key)
        return jax.random.split(k, n).squeeze()

    def save(self, path):
        raise NotImplementedError()

    @classmethod
    def load(cls, path):
        raise NotImplementedError()
