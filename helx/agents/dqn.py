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

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import rlax
from chex import Array
from flax import linen as nn
from flax import struct
from jax.random import KeyArray
from optax import GradientTransformation, OptState

from helx.environment.base import Environment
from helx.mdp import Action
from helx.spaces import Discrete

from ..mdp import Trajectory, Transition
from ..memory import ReplayBuffer
from ..networks import EGreedyHead, QHead
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


class DQN(Agent):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    hparams: DQNHparams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)
    iteration: int = struct.field(pytree_node=False)
    actor: nn.Module = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)
    params_actor: nn.FrozenDict = struct.field(pytree_node=False)
    params_critic: nn.FrozenDict
    params_target: nn.FrozenDict
    opt_state: OptState
    memory: ReplayBuffer

    @classmethod
    def create(cls, hparams, optimiser, key, representation_net):
        # config
        assert isinstance(hparams.action_space, Discrete)
        iteration = 0
        actor = EGreedyHead(
            hparams.replay_start,
            hparams.initial_exploration,
            hparams.final_exploration,
            hparams.final_exploration_frame,
        )
        critic = nn.Sequential([representation_net, QHead(hparams.action_space)])
        sample_output, params_critic = critic.init_with_output(
            key, hparams.obs_space.sample(key), hparams.action_space.sample(key)
        )
        _, params_actor = actor.init_with_output(
            key, *(sample_output, key, iteration, 1, False)
        )
        params_target = params_critic.copy({})  # type: ignore
        example_item = (
            hparams.obs_space.sample(key),
            hparams.action_space.sample(key),
            0.0,
            hparams.obs_space.sample(key),
            False,
        )
        memory = ReplayBuffer.create(example_item, hparams.replay_memory_size)
        opt_state = optimiser.init(params_critic)

        agent = cls(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            params_critic=params_critic,  # type: ignore
            params_actor=params_actor,  # type: ignore
            params_target=params_target,
            opt_state=opt_state,
            iteration=iteration,
            memory=memory,
        )

        return agent

    def sample_action(self, env: Environment, key: KeyArray, eval: bool = False) -> Action:
        n_actions = env.n_parallel()
        q_values = self.critic.apply(self.params_critic, env.state())
        actions, _ = self.actor.apply(
            self.params_actor, q_values=q_values, key=key, iteration=self.iteration, n_actions=n_actions, eval=eval
        )
        return actions

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
    ) -> Tuple[Array, Any]:
        s_tm1, a_tm1, r_t, s_t, d = transition
        q_tm1 = jnp.asarray(self.critic.apply(params, s_tm1))
        q_t = jnp.asarray(self.critic.apply(self.params_target, s_t))

        td_error = jnp.asarray(rlax.q_learning(q_tm1, a_tm1, r_t, d, q_t))
        loss = rlax.l2_loss(td_error).mean()
        return loss, ()

    def update(self, episode: Trajectory, key: KeyArray) -> Tuple[Agent, Dict[str, Any]]:
        # update iteration
        iteration = self.iteration + 1

        # update memory
        transitions = episode.transitions()
        buffer = self.memory.add_range(transitions)

        # log data after update
        log = {}
        log.update({"Iteration": iteration})
        log.update({"train/Return": jnp.sum(episode.r)})
        log.update({"Buffer size": buffer.size()})

        # if replay buffer is not big enough, or it's not an update step, there is nothing else to do
        if buffer.size() < self.hparams.replay_start:
            return self, log
        if iteration % self.hparams.update_frequency != 0:
            return self, log

        # update dqn state
        buffer, episode_batch = buffer.sample(key=key, n=self.hparams.batch_size)
        critic_params, opt_state, loss, _ = self.sgd_step(
            self.params_critic,
            episode_batch,
            self.opt_state,
        )
        self.critic_params = critic_params
        self.opt_state = opt_state
        self.params_target = rlax.periodic_update(
            self.critic_params,
            self.params_target,
            jnp.asarray(self.iteration),
            self.hparams.target_network_update_frequency,
        )

        # and log the loss
        log.update({"train/total_loss": loss})
        return self, log
