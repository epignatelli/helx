from __future__ import annotations

from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from chex import Array
from jax.random import KeyArray

from ..mdp import Action

from .modules import Identity


class Actor(nn.Module):
    policy_head: nn.Module
    representation_net: nn.Module = Identity()

    def __call__(self, observation: Array, key: KeyArray) -> Tuple[Action, Array]:
        representation = self.representation_net(observation)
        return self.policy_head(representation, key)


class EGreedyPolicy(nn.Module):
    initial_exploration_frame: int
    initial_exploration: float
    final_exploration: float
    final_exploration_frame: int

    @nn.compact
    def __call__(self, q_values: Array, key, **kwargs) -> Tuple[Action, Array]:
        eval = kwargs.pop("eval", False)
        x = self.variable("stats", "iteration", lambda: jnp.ones((), dtype=jnp.float32))

        eps = jnp.interp(
            x.value,
            jnp.asarray([self.initial_exploration_frame, self.final_exploration_frame]),
            jnp.asarray([self.initial_exploration, self.final_exploration]),
        )
        eps = jax.lax.select(eval, 0.0, eps)
        x.value += 1

        distr = distrax.EpsilonGreedy(q_values, eps)  # type: ignore
        action, log_probs = stop_gradient(distr.sample_and_log_prob(seed=key))
        return action, log_probs


class GaussianPolicy(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, representation: Array, key: KeyArray) -> Tuple[Action, Array]:
        mu = nn.Dense(features=self.action_size)(representation)
        log_sigma = nn.Dense(features=self.action_size)(representation)
        # reparametrisation trick: sample from normal and multiply by std
        noise = jax.random.normal(key, (self.action_size,))
        action = jnp.tanh(mu + log_sigma * noise)
        log_probs = distrax.Normal(loc=mu, scale=jnp.exp(log_sigma)).log_prob(action)
        return action, log_probs


class SoftmaxPolicy(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, representation: Array, key: KeyArray) -> Tuple[Action, Array]:
        logits = nn.Dense(features=self.n_actions)(representation)
        action, log_probs = distrax.Softmax(logits).sample_and_log_prob(seed=key)
        return action, log_probs
