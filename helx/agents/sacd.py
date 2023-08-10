# Copyright 2023 The Helx Authors.
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

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.core.scope import VariableDict as Params
from jax import Array
from jax.random import KeyArray

from ..modules import Temperature, Split, Parallel
from ..mdp import Timestep, TERMINATION
from ..spaces import Discrete
from .sac import SAC, SACHParams, SACState


class SACDHParams(SACHParams):
    action_space: Discrete


class SACD(SAC):
    """Soft Actor-Critic agent with softmax policy for discrete action spaces
    and a twinned Q-network for value approximation with a separate target network
    updated by polyak averaging the online network parameters.
    See:
        1. Haarnoja T., 2018 (https://arxiv.org/abs/1801.01290)
            for the base SAC
        2. Haarnoja T., 2019 (https://arxiv.org/abs/1812.05905)
            for the tunable entropy temperature and the ablation of the value network
        3. Christodoulou P., 2019 (https://arxiv.org/abs/1910.07207)
            for the adaptation to discrete action spaces
    """

    @classmethod
    def create(
        cls,
        hparams: SACDHParams,
        optimiser: optax.GradientTransformation,
        actor_backbone: nn.Module,
        critic_backbone: nn.Module,
    ) -> SACD:
        n_actions = hparams.action_space.maximum
        actor = nn.Sequential([actor_backbone, nn.Dense(n_actions)])
        critic = nn.Sequential(
            [
                Split(2),
                Parallel((critic_backbone, jtu.tree_map(lambda x: x, critic_backbone))),
                Parallel((nn.Dense(n_actions), nn.Dense(n_actions))),
            ]
        )
        temperature = Temperature()
        return SACD(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            temperature=temperature,
        )

    def sample_action(
        self, train_state: SACState, obs: Array, *, key: KeyArray, eval: bool = False
    ) -> Array:
        params_actor, _, _ = train_state.params
        logits = jnp.asarray(self.actor.apply(params_actor, obs))
        action = jax.random.categorical(key, logits)
        return action

    def loss(
        self,
        params: Tuple[Params, Params, Params],
        transition: Timestep,
    ) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
        s_tm1 = transition.observation[:-1]
        s_t = transition.observation[1:]
        r_t = transition.reward[:-1][0]  # [0] because scalar
        terminal_tm1 = transition.step_type[:-1] != TERMINATION

        # params
        batch_matmul = lambda a, b: jnp.sum(a * b, axis=-1, keepdims=True)
        params_actor, params_critic, params_temperature = params

        # current estimates
        temperature = self.temperature(params_temperature)
        alpha = jax.lax.stop_gradient(temperature)
        logits_a_tm1 = self.actor(params_actor, s_tm1)
        logits_a_t = self.actor(params_actor, s_t)
        logprobs_a_tm1 = jax.nn.log_softmax(logits_a_tm1)
        logprobs_a_t = jax.nn.log_softmax(logits_a_t)
        probs_a_tm1 = jnp.exp(logprobs_a_tm1)

        # augment reward with policy entropy
        policy_entropy = -jnp.sum(probs_a_tm1 * logprobs_a_tm1, axis=-1)
        # SACLite: https://arxiv.org/abs/2201.12434
        policy_entropy = policy_entropy * self.hparams.entropy_rewards
        r_t = r_t + alpha * policy_entropy

        # critic is a double Q network
        qA_tm1, qB_tm1 = self.critic(params_critic, s_tm1)
        qA_t, qB_t = self.critic(params_critic, s_t)

        # actor_target: $V(s_t) = \pi(s_t)^T \cdot [\alpha \log(\pi(s_t)) + Q(s_t)]$
        q_tm1 = jax.lax.stop_gradient(
            jnp.min(jnp.stack([qA_tm1, qB_tm1], axis=0), axis=0)
        )
        actor_loss = batch_matmul(probs_a_tm1, alpha * logprobs_a_tm1 - q_tm1)

        # critic target: $V(s_{t+1}) = \pi(s_{t+1})^T \cdot [Q(s_{t+1}) - \alpha \log(\pi(s_{t+1}))]$
        probs_a_t = jnp.exp(logprobs_a_t)
        q_t = jnp.min(jnp.stack([qA_t, qB_t], axis=0), axis=0)
        # SAC calculates the expectation over q-values by averaging
        # over the iterations of the training.
        # Here we have the q and action probs for all actions
        # so we calculate the state-value as the expectation of the q-values
        v_1 = batch_matmul(probs_a_t, (q_t - alpha * logprobs_a_t))
        q_target = jax.lax.stop_gradient(
            r_t + terminal_tm1 * self.hparams.discount * v_1
        )
        critic_loss = optax.l2_loss(qA_tm1, q_target) + optax.l2_loss(qB_tm1, q_target)
        critic_loss = jnp.asarray(critic_loss)

        # temperature target:$ H(\pi(s_t)) = 0.98 * -\sum_a \pi(a|s_t) \log(\pi(a|s_t))$
        target_entropy = 0.98 * (-jnp.log(1 / self.hparams.action_space.ndim))
        logprobs_a_tm1 = jax.lax.stop_gradient(logprobs_a_tm1)
        probs_a_tm1 = jax.lax.stop_gradient(probs_a_tm1)
        temperature_loss = -(temperature * logprobs_a_tm1 + target_entropy)
        temperature_loss = batch_matmul(probs_a_tm1, temperature_loss)

        loss = actor_loss + critic_loss + temperature_loss
        aux = (actor_loss, critic_loss, temperature_loss, policy_entropy, alpha)
        return loss, aux
