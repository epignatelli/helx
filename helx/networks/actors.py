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


from __future__ import annotations

import operator
from functools import reduce
from typing import Tuple
import chex

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array, Shape
from jax.random import KeyArray
import optax

from ..mdp import Action
from .modules import Identity


class Actor(nn.Module):
    policy_head: nn.Module
    representation_net: nn.Module = Identity()

    def __call__(self, observation: Array, key: KeyArray) -> Tuple[Action, Array]:
        representation = self.representation_net(observation)
        return self.policy_head(representation, key)


class EGreedyHead(nn.Module):
    eps_start: int
    eps_init_value: float
    eps_end_value: float
    eps_end: int

    @nn.compact
    def __call__(
        self,
        q_values: Array,
        key: KeyArray,
        iteration: int,
        n_actions: int = 1,
        eval: bool = False,
    ) -> Tuple[Action, Array]:
        """Sample actions from an epsilon-greedy policy, and returns
        both the sampled actions and their log probabilities."""
        # anneal epsilon
        eps = optax.polynomial_schedule(
            self.eps_init_value,
            self.eps_end_value,
            1,
            self.eps_end - self.eps_start,
            self.eps_start,
        )(iteration)

        # no epsilon during evaluation
        eps = jax.lax.select(eval, 0.0, eps)
        distr = distrax.EpsilonGreedy(q_values, eps)  # type: ignore
        actions, log_probs = distr.sample_and_log_prob(seed=key, sample_shape=(n_actions,))
        return actions, log_probs


class GaussianHead(nn.Module):
    action_shape: Shape  # type: ignore

    @nn.compact
    def __call__(self, representation: Array, key: KeyArray) -> Tuple[Action, Array]:
        action_size = reduce(operator.mul, self.action_shape, 1)
        mu = nn.Dense(features=action_size)(representation)
        log_sigma = nn.Dense(features=action_size)(representation)

        # reparametrisation trick: sample from normal and multiply by std
        noise = jax.random.normal(key, (action_size,))
        action = jnp.tanh(mu + log_sigma * noise)
        log_prob = distrax.Normal(loc=mu, scale=jnp.exp(log_sigma)).log_prob(action)

        action = action.reshape(self.action_shape)
        log_prob = log_prob.reshape(self.action_shape)

        chex.assert_shape(action, self.action_shape)
        chex.assert_shape(log_prob, self.action_shape)

        return action, log_prob


class SoftmaxHead(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, representation: Array, key: KeyArray) -> Tuple[Action, Array]:
        logits = nn.Dense(features=self.n_actions)(representation)

        action = distrax.Softmax(logits).sample(seed=key)
        log_probs = jax.nn.log_softmax(logits)

        chex.assert_shape(action, ())
        chex.assert_shape(log_probs, (self.n_actions,))

        return action, log_probs
