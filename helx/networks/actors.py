from __future__ import annotations

from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array, PyTreeDef
from jax.random import KeyArray

from .modules import Identity


class Actor(nn.Module):
    policy_head: nn.Module
    representation_net: nn.Module = Identity()

    def __call__(self, observation: Array, key: KeyArray) -> PyTreeDef | Array:
        representation = self.representation_net(observation)
        return self.policy_head(representation, key)


class GaussianPolicy(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, observation: Array, key: KeyArray) -> Tuple[Array, Array]:
        mu = nn.Dense(features=self.action_size)(observation)
        log_sigma = nn.Dense(features=self.action_size)(observation)
        # reparametrisation trick
        noise = jax.random.normal(key, (self.action_size,))
        action = jnp.tanh(mu + log_sigma * noise)
        log_probs = distrax.Normal(loc=mu, scale=jnp.exp(log_sigma)).log_prob(action)
        return action, log_probs


class SoftmaxPolicy(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, observation: Array, key: KeyArray) -> Tuple[Array, Array]:
        logits = nn.Dense(features=self.n_actions)(observation)
        action, log_probs = distrax.Softmax(logits).sample_and_log_prob(seed=key)
        return action, log_probs
