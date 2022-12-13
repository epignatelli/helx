"""Base class for a Reinforcement Learning agent."""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import abc
from typing import Any, NamedTuple, Tuple

import jax
from chex import Array, PyTreeDef
from dm_env import Environment

from ..mdp import Episode


class Hparams(NamedTuple):
    ...


class Agent():
    def __init__(self):
        self.loss = jax.jit(self.loss)

        if not hasattr(self, "hparams"):
            self.hparams = Hparams()
        if not hasattr(self, "iteration"):
            self.iteration = 0

    @abc.abstractmethod
    def loss(
        self, params: PyTreeDef, sarsa: Episode, *args, **kwargs
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
    def policy(
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

    def unroll(
        self, env: Environment, eval: bool = False, max_steps: int = int(2e9)
    ) -> Episode:
        """Deploys the agent in the environment for a full episode.
            In case of batched environments, the each property of the episode
            has an additional `batch` axis at index `0`.
            The episode terminates only for absorbing states, and the number of
            maximum steps can be specified in the environment itself.
            While using `experiment.run` the result of this function is passed
            to the `update` method.
        Args:
            env (dm_env.Environment): the environment to interact with implements
                the `dm_env.Environment` interface, and NOT a `gym` interface.
                You can wrap the former around the latter using  the
                `bsuite.utils.gym_wrapper.DMFromGymWrapper` wrapper.
            eval (bool): eval flag passed to the `policy` method.
        Returns:
            (Episode): an full episode following the current policy.
                Each property of the episode has an additional `batch` axis at index `0`.
                in case of batched environments.
        """
        t = 0
        timestep = env.reset()
        episode = Episode.start(timestep)
        while (not timestep.last()) and t < max_steps:
            t += 1
            action, _ = self.policy(timestep.observation, eval=eval)
            timestep = env.step(action.item())
            episode.add(timestep, action)
        return episode

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError
