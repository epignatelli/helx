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

from typing import Any, Dict

import jax
import jax.numpy as jnp
import rlax
from flax import linen as nn
from optax import GradientTransformation

from ..mdp import Trajectory, Transition
from ..memory import Buffer
from ..networks import (
    Actor,
    AgentNetwork,
    QHead,
    GaussianHead,
    SoftmaxHead,
    Temperature,
)
from .. import spaces
from .agent import Agent, Hparams


class A2CHparams(Hparams):
    # network
    action_space: spaces.Space
    tau: float = 0.005
    # rl
    discount: float = 0.99
    n_steps: int = 1
    lambda_td: float = 1.0
    entropy_coefficient: float = 0.01
    # sgd
    batch_size: int = 32
    learning_rate: float = 3e-4


class A2C(Agent[A2CHparams]):
    """Synchronous version of Advantage Actor Critic (A2C) with entropy regularisation,
    as described in
    "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016),
    https://arxiv.org/abs/1602.01783.
    """

    def __init__(
        self,
        hparams: A2CHparams,
        optimiser: GradientTransformation,
        seed: int,
        actor_network: nn.Module,
        critic_network: nn.Module,
    ):
        """
        Initialises the agent.
        Args:
            hparams (A2CHparams): the hyperparameters for the agent.
            optimiser (GradientTransformation): an optax optimiser to perform gradient descent.
            seed (int): the seed to use for the agent.
                Using the same seed will result in reproducible agent results.
            actor_net: the network to use for the actor *without the policy head*.
                The policy head will be added automatically to match the action space provided in `hparams`.
            critic_net: the network to use for the critic *without the Q head*.
                The Q head will be added automatically to match the action space provided in `hparams`.
        """
        if spaces.is_discrete(hparams.action_space):
            policy_head = SoftmaxHead(n_actions=hparams.action_space.shape[0])
        else:
            policy_head = GaussianHead(action_shape=hparams.action_space.shape)

        network = AgentNetwork(
            actor_net=Actor(
                representation_net=actor_network,
                policy_head=policy_head,
            ),
            critic_net=QHead(
                n_actions=1,
                representation_net=critic_network,
            ),
            extra_net=Temperature(),
        )

        super().__init__(hparams, network, optimiser, seed)
        self.memory = Buffer(hparams.batch_size, seed)

    def loss(
        self,
        params: nn.FrozenDict,
        transition_batch: Transition,
        key: jax.random.KeyArray,
    ):
        s_t, a_t, r_tp1, s_tp1, _ = transition_batch

        # critic loss
        q_t = jnp.asarray(self.network.critic(params, s_t))
        q_tp1 = jnp.asarray(self.network.critic(params, s_tp1))
        gamma_t = jnp.broadcast_to(self.hparams.discount, q_tp1.shape)
        lambda_ = jnp.broadcast_to(self.hparams.lambda_td, q_tp1.shape)
        g_t = rlax.lambda_returns(r_tp1, gamma_t, q_tp1, lambda_, stop_target_gradients=True)
        critic_loss = rlax.l2_loss(q_t, g_t)

        # actor loss
        a_t, logits_t = self.network.actor(params, s_t, key)
        adv_t = q_t - g_t
        w_t =  jnp.broadcast_to(1.0, adv_t.shape)
        actor_loss = rlax.policy_gradient_loss(logits_t, a_t, adv_t, w_t, use_stop_gradient=True)

        # entropy loss
        entropy_loss = rlax.entropy_loss(logits_t, w_t)

        aux = (actor_loss, critic_loss, entropy_loss)
        total_loss = sum(aux)  # type: ignore
        return total_loss, aux

    def update(self, episode: Trajectory) -> Dict[str, Any]:
        # update iteration
        self.iteration += 1
        log = {}
        log.update({"Iteration": self.iteration})
        log.update({"train/Return": episode.r.sum()})

        # check if the batch is full
        if not self.memory.full():
            return log

        # if it is, perform a learning step
        episode_batch: Transition = self.memory.sample(self.hparams.batch_size)
        params, opt_state, loss, aux = self.sgd_step(
            self.params,
            episode_batch,
            self.opt_state,
            self._new_key(),
        )
        self.opt_state = opt_state
        self.params = params

        # and log the update
        aux = jax.tree_map(jnp.mean, aux)  # reduce aux
        actor_loss, critic_loss, entropy_loss = aux
        log.update({"train/total_loss": loss})
        log.update({"train/actor_loss": actor_loss})
        log.update({"train/critic_loss": critic_loss})
        log.update({"train/entropy_loss": entropy_loss})
        return log
