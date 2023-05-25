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
from functools import partial

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import rlax
from chex import Array, dataclass
from flax import linen as nn
from jax.random import KeyArray
from optax import GradientTransformation

from helx.environment.base import Environment
from helx.mdp import Action
from helx.spaces import Discrete

from ..mdp import Trajectory, Transition
from ..memory import Buffer
from ..networks import EGreedyHead, QHead
from .agent import Agent, AgentState, Hparams


@dataclass
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


@dataclass
class DQNState(AgentState):
    iteration: int
    params_critic: nn.FrozenDict
    params_actor: nn.FrozenDict
    opt_state: Any
    key: KeyArray
    params_target: nn.FrozenDict
    memory: Buffer[Transition]


@dataclass
class DQN(Agent[DQNHparams, DQNState]):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    hparams: DQNHparams
    optimiser: GradientTransformation
    seed: int
    actor: nn.Module
    critic: nn.Module

    @classmethod
    def init(cls, hparams, optimiser, seed, representation_net):
        # config
        assert isinstance(hparams.action_space, Discrete)
        actor = EGreedyHead(
                hparams.replay_start,
                hparams.initial_exploration,
                hparams.final_exploration,
                hparams.final_exploration_frame,
            )
        critic = nn.Sequential([representation_net, QHead(hparams.action_space)])

        # state
        iteration = 0
        key = jax.random.PRNGKey(seed)
        params_critic, sample_output = critic.init_with_output(
            key, hparams.obs_space.sample(key), hparams.action_space.sample(key)
            )
        params_actor, _ = actor.init_with_output(
            key, sample_output, key, iteration, 1, False
            )
        memory = Buffer[Transition](hparams.replay_memory_size)
        params_target = params_critic.copy({})
        opt_state = optimiser.init(params_critic)

        agent = cls(hparams=hparams, optimiser=optimiser, seed=seed, actor=actor, critic=critic)
        state = DQNState(
            iteration=iteration,
            params_actor=params_actor,
            params_critic=params_critic,
            opt_state=opt_state,
            key=key,
            params_target=params_target,
            memory=memory,
        )
        return (agent, state)

    def sample_action(self, state: DQNState, env: Environment, eval: bool = False) -> Action:
        n_actions = env.n_parallel()
        q_values = self.critic.apply(state.params_critic, env.state())
        actions, _ = self.actor.apply(state.params_actor, q_values, state.iteration, eval, n_actions=n_actions)
        return actions

    def loss(
        self,
        state: DQNState,
        transition: Transition,
    ) -> Tuple[Array, Any]:
        s_tm1, a_tm1, r_t, s_t, d = transition

        q_tm1 = jnp.asarray(self.critic.apply(state.params_critic, s_tm1))
        q_t = jnp.asarray(self.critic.apply(state.params_target, s_t))

        td_error = jnp.asarray(rlax.q_learning(q_tm1, a_tm1, r_t, d, q_t))
        loss = rlax.l2_loss(td_error).mean()
        return loss, ()

    @partial(jax.jit, static_argnums=(0,))
    def update(self, state, episode: Trajectory) -> Tuple[Any, Dict[str, Any]]:
        # update iteration
        iteration = state.iteration + 1

        # update memory
        transitions: List[Transition] = episode.transitions()
        state.memory.add_range(transitions)

        # log data after update
        log = {}
        log.update({"Iteration": iteration})
        log.update({"train/Return": jnp.sum(episode.r)})
        log.update({"Buffer size": len(state.memory)})

        # if replay buffer is not big enough, or it's not an update step, there is nothing else to do
        if (len(state.memory) < self.hparams.replay_start) or (
            iteration % self.hparams.update_frequency != 0
        ):
            state = (iteration, params, opt_state, params_target, memory)
            return state, log

        # update dqn state
        episode_batch: Transition = memory.sample(self.hparams.batch_size)
        critic_params, opt_state, loss, _ = self.sgd_step(
            params,
            episode_batch,
            opt_state,
            params_target,
        )
        opt_state = opt_state
        params_target = rlax.periodic_update(
            params,
            params_target,
            jnp.asarray(iteration),
            self.hparams.target_network_update_frequency,
        )

        state = DQNState(iteration=iteration, critic_params=, actor_params opt_state, params_target, memory)
        # and log the loss

        log.update({"train/total_loss": loss})
        return state, log
