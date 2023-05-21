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

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import rlax
from chex import Array
from flax import linen as nn
from optax import GradientTransformation

from helx.networks.modules import Identity

from helx.spaces import Discrete

from ..mdp import Trajectory, Transition
from ..memory import Buffer
from ..networks import AgentNetwork, EGreedyHead, QHead
from .agent import Agent, Hparams


class DQNHparams(Hparams):
    # network
    hidden_size: int = 128
    # rl
    initial_exploration: float = 1.0
    final_exploration: float = 0.01
    final_exploration_frame: int = 1000000
    replay_start: int = 1000
    replay_memory_size: int = 1000
    update_frequency: int = 1
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    n_steps: int = 1
    # sgd
    batch_size: int = 32
    learning_rate: float = 0.00025
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01


class DQN(Agent[DQNHparams]):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    def __init__(
        self,
        hparams: DQNHparams,
        optimiser: GradientTransformation,
        seed: int,
        representation_net: nn.Module,
    ):
        assert isinstance(hparams.action_space, Discrete)
        n_actions = hparams.action_space.n_bins
        critic_head = QHead(n_actions, representation_net)
        actor_head = EGreedyHead(
            hparams.replay_start,
            hparams.initial_exploration,
            hparams.final_exploration,
            hparams.final_exploration_frame,
        )
        network = AgentNetwork(
            shared_net=critic_head,
            actor_net=actor_head,
            critic_net=Identity(),
        )
        super().__init__(hparams, network, optimiser, seed)
        self.memory = Buffer[Transition](hparams.replay_memory_size)
        self.params_target = self.params.copy({})

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        params_target: nn.FrozenDict,
    ) -> Tuple[Array, Any]:
        s_0, a_0, r_1, s_1, d = transition
        q_0 = self.network.critic(params, s_0)
        q_1 = self.network.critic(params_target, s_1)

        # ignoring type because flax modules can return anything
        # we should either strongly type the network or cast the output
        q_1 = jax.lax.stop_gradient(jnp.max(q_1))  # type: ignore
        td_error = r_1 + (1 - d) * self.hparams.discount * q_1 - q_0[a_0]  # type: ignore
        loss = jnp.mean(rlax.l2_loss(td_error))
        return loss, ()

    def update(self, episode: Trajectory) -> Dict[str, Any]:
        # increment iteration
        self.iteration += 1
        log = {}

        # update memory
        transitions: List[Transition] = episode.transitions()
        self.memory.add_range(transitions)

        log.update({"Iteration": self.iteration})
        log.update({"train/Return": jnp.sum(episode.r)})
        log.update({"Buffer size": len(self.memory)})

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.memory) < self.hparams.replay_start:
            return log

        # update every `update_frequency` steps
        if self.iteration % self.hparams.update_frequency != 0:
            return log

        # update dqn state
        episode_batch: Transition = self.memory.sample(self.hparams.batch_size)
        params, opt_state, loss, _ = self.sgd_step(
            self.params,
            episode_batch,
            self.opt_state,
            self.params_target,
        )
        self.opt_state = opt_state
        self.params = params
        self.params_target = rlax.periodic_update(
            self.params,
            self.params_target,
            jnp.asarray(self.iteration),
            self.hparams.target_network_update_frequency,
        )

        # and log the loss
        log.update({"train/total_loss": loss})
        return log
