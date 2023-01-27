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

import jax
import jax.numpy as jnp
import rlax
from chex import Array
from flax import linen as nn
from jax.lax import stop_gradient
from optax import GradientTransformation

import wandb

from ..mdp import Episode, Transition
from ..memory import ReplayBuffer
from ..networks import (
    Actor,
    AgentNetwork,
    DoubleQCritic,
    GaussianPolicy,
    Temperature,
    deep_copy,
)
from ..spaces import Continuous
from .agent import Agent, Hparams


class SACHparams(Hparams):
    # network
    action_space: Continuous
    tau: float = 0.005
    # rl
    replay_start: int = 1000
    replay_memory_size: int = 1000
    update_frequency: int = 1
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    n_steps: int = 1
    entropy_rewards: bool = False
    # sgd
    batch_size: int = 32
    learning_rate: float = 3e-4


class SAC(Agent[SACHparams]):
    """Implements Soft Actor-Critic with gaussian policy for continuous action spaces.
    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep
        reinforcement learning with a stochastic actor."
        International conference on machine learning. PMLR, 2018.
        https://arxiv.org/abs/1801.01290 for the base SAC.
    Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
        arXiv preprint arXiv:1812.05905 (2018).
        https://arxiv.org/abs/1812.05905 for the tunable entropy temperature
    Haonanm Yu, et al. "Do you need reward entropy (in practice)?".
        arXiv preprint arXiv:2201.12434. 2022 Jan 28.
        https://arxiv.org/abs/2201.12434

    Args:
        network (AgentNetwork): the network to use for the agent.
            the `actor_net` is usually a `GaussianPolicy` for continuous action spaces.
            the `critic_net` is double Q network.
            the `extra_net` is usually a `Temperature` layer to automatically
            tune the entropy pull.
        optimiser (GradientTransformation): an optax optimiser to perform gradient descent.
        hparams (SACHparams): the hyperparameters for the agent.
        seed (int): the seed to use for the agent.
            Using the same seed will result in reproducible agent results.
    """

    def __init__(
        self,
        hparams: SACHparams,
        optimiser: GradientTransformation,
        seed: int,
        actor_representation_net: nn.Module,
        critic_representation_net: nn.Module,
    ):
        network = AgentNetwork(
            actor_net=Actor(
                representation_net=actor_representation_net,
                policy_head=GaussianPolicy(action_shape=hparams.action_space.shape),
            ),
            critic_net=DoubleQCritic(
                n_actions=1,
                representation_net_a=deep_copy(critic_representation_net),
                representation_net_b=deep_copy(critic_representation_net),
            ),
            extra_net=Temperature(),
        )

        super().__init__(hparams, network, optimiser, seed)
        self.memory = ReplayBuffer(hparams.replay_memory_size)
        self.params_target: nn.FrozenDict = self.params.copy({})
        self.dim_actions = hparams.action_space.shape[0]

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        key: jax.random.KeyArray,
        params_target: nn.FrozenDict,
    ):
        s_0, _, r_1, s_1, d = transition

        # current estimates
        temperature = self.network.extra(params)
        alpha = stop_gradient(temperature)
        _, logprobs_a_0 = self.network.actor(params, s_0, key)
        _, logprobs_a_1 = self.network.actor(params, s_1, key)
        probs_a_0 = jnp.exp(logprobs_a_0)
        # critic is a double Q network
        qA_0, qB_0 = self.network.critic(params, s_0)  # type: ignore
        qA_1, qB_1 = self.network.critic(params_target, s_1)  # type: ignore

        # augment reward with policy entropy
        policy_entropy = -jnp.sum(probs_a_0 * logprobs_a_0, axis=-1)
        # SACLite: https://arxiv.org/abs/2201.12434
        policy_entropy = policy_entropy * self.hparams.entropy_rewards
        r_1 = r_1 + alpha * policy_entropy

        # actor loss
        q_0 = stop_gradient(jnp.min(jnp.stack([qA_0, qB_0], axis=0), axis=0))
        actor_loss = alpha * logprobs_a_0 - q_0

        # critic losses
        q_1 = jnp.min(jnp.stack([qA_1, qB_1], axis=0), axis=0)
        v_1 = q_1 - alpha * logprobs_a_1
        q_target = stop_gradient(r_1 + (1 - d) * self.hparams.discount * v_1)
        critic_loss = rlax.l2_loss(qA_0, q_target) + rlax.l2_loss(qB_0, q_target)

        # temperature loss
        target_entropy = -self.hparams.action_space.n_dim
        logprobs_a_0 = stop_gradient(logprobs_a_0)
        temperature_loss = -(temperature * (logprobs_a_0 + target_entropy))

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
