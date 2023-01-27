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


# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
import rlax
from chex import Array
from jax.lax import stop_gradient
from optax import GradientTransformation

import wandb
from helx.spaces import Discrete

from ..mdp import Episode, Transition
from ..memory import ReplayBuffer
from ..networks import (
    Actor,
    AgentNetwork,
    DoubleQCritic,
    SoftmaxPolicy,
    Temperature,
    deep_copy,
)
from .agent import Agent
from .sac import SACHparams


class SACDHparams(SACHparams):
    action_space: Discrete


class SACD(Agent[SACDHparams]):
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

    def __init__(
        self,
        hparams: SACDHparams,
        optimiser: GradientTransformation,
        seed: int,
        actor_representation_net: nn.Module,
        critic_representation_net: nn.Module,
    ):
        network = AgentNetwork(
            actor_net=Actor(
                representation_net=actor_representation_net,
                policy_head=SoftmaxPolicy(n_actions=hparams.action_space.n_bins),
            ),
            critic_net=DoubleQCritic(
                n_actions=hparams.action_space.n_bins,
                representation_net_a=deep_copy(critic_representation_net),
                representation_net_b=deep_copy(critic_representation_net),
            ),
            extra_net=Temperature(),
        )

        super().__init__(hparams, network, optimiser, seed)
        self.memory = ReplayBuffer(hparams.replay_memory_size)
        self.params_target: nn.FrozenDict = self.params.copy({})

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        keys: jax.random.KeyArray,
        params_critic_target,
    ):
        s_0, _, r_1, s_1, d = transition
        batch_matmul = lambda a, b: jnp.sum(a * b, axis=-1, keepdims=True)

        # current estimates
        temperature = self.network.extra(params)
        alpha = stop_gradient(temperature)
        _, logprobs_a_0 = self.network.actor(params, s_0, keys)
        _, logprobs_a_1 = self.network.actor(params, s_1, keys)
        probs_a_0 = jnp.exp(logprobs_a_0)
        # critic is a double Q network
        qA_0, qB_0 = self.network.critic(params, s_0)  # type: ignore
        qA_1, qB_1 = self.network.critic(params_critic_target, s_1)  # type: ignore

        # augment reward with policy entropy
        policy_entropy = -jnp.sum(probs_a_0 * logprobs_a_0, axis=-1)
        # SACLite: https://arxiv.org/abs/2201.12434
        policy_entropy = policy_entropy * self.hparams.entropy_rewards
        r_1 = r_1 + alpha * policy_entropy

        # actor target: V(sₜ) = π(sₜ)ᵀ • [αlog(π(sₜ)) + Q(sₜ)]
        q_0 = stop_gradient(jnp.min(jnp.stack([qA_0, qB_0], axis=0), axis=0))
        actor_loss = batch_matmul(probs_a_0, alpha * logprobs_a_0 - q_0)

        # critic target: V(sₜ₊₁) = π(sₜ₊₁)ᵀ • [Q(sₜ₊₁) - αlog(π(sₜ₊₁))]
        probs_a_1 = jnp.exp(logprobs_a_1)
        q_1 = jnp.min(jnp.stack([qA_1, qB_1], axis=0), axis=0)
        # compared to SAC, which takes the expectation over the action
        # because it doesn't have the probs over all possible actions
        # here we can, so calculate the exact value of the state
        v_1 = batch_matmul(probs_a_1, (q_1 - alpha * logprobs_a_1))
        q_target = stop_gradient(r_1 + (1 - d) * self.hparams.discount * v_1)
        critic_loss = rlax.l2_loss(qA_0, q_target) + rlax.l2_loss(qB_0, q_target)

        #
        target_entropy = 0.98 * (-jnp.log(1 / self.hparams.action_space.n_bins))
        logprobs_a_0 = stop_gradient(logprobs_a_0)
        probs_a_0 = stop_gradient(probs_a_0)
        temperature_loss = -(temperature * logprobs_a_0 + target_entropy)
        temperature_loss = batch_matmul(probs_a_0, temperature_loss)

        loss = actor_loss + critic_loss + temperature_loss
        aux = (actor_loss, critic_loss, temperature_loss, policy_entropy, alpha)
        return loss, aux

    def update(self, episode: Episode) -> Array:
        # update iteration
        self.iteration += 1
        wandb.log({"Iteration": self.iteration})

        # update memory
        transitions: List[Transition] = episode.transitions()
        self.memory.add_range(transitions)
        wandb.log({"Buffer size": len(self.memory)})

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.memory) < self.hparams.replay_start:
            return jnp.asarray([])

        # check update period
        if self.iteration % self.hparams.update_frequency != 0:
            return jnp.asarray([])

        episode_batch: Transition = self.memory.sample(self.hparams.batch_size)
        params, opt_state, loss, aux = self.sgd_step(
            self.params,
            episode_batch,
            self.opt_state,
            self._new_key(),
            self.params_target,
        )

        # update dqn state
        self.opt_state = opt_state
        self.params = params
        self.params_target = jax.tree_map(
            lambda theta, theta_: theta * self.hparams.tau
            + (1 - self.hparams.tau) * theta_,
            self.params,
            self.params_target,
        )

        aux = jax.tree_map(jnp.mean, aux)  # reduce aux
        actor_loss, critic_loss, temperature_loss, policy_entropy, alpha = aux
        wandb.log({"train/total_loss": loss.item()})
        wandb.log({"train/actor_loss": actor_loss.item()})
        wandb.log({"train/critic_loss": critic_loss.item()})
        wandb.log({"train/temperature_loss": temperature_loss.item()})
        wandb.log({"train/policy_entropy": policy_entropy.item()})
        wandb.log({"train/alpha": alpha.item()})
        wandb.log({"train/Return": jnp.sum(episode.r).item()})  # type: ignore
        return loss
