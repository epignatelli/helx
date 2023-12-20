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

import distrax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import rlax
from flax import linen as nn
from flax import struct
from flax.core.scope import VariableDict as Params
from jax import Array
from jax.random import KeyArray
from optax import GradientTransformation

from helx.base.mdp import TERMINATION, Timestep
from helx.base.memory import ReplayBuffer
from helx.base.spaces import Discrete
from helx.base import losses
from .agent import Agent, HParams, Log, AgentState


class DQNHParams(HParams):
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


class DQNLog(Log):
    buffer_size: Array = jnp.asarray(0)
    critic_loss: Array = jnp.asarray(float("inf"))


class DQNState(AgentState):
    params: Params = struct.field(pytree_node=True)
    params_target: Params = struct.field(pytree_node=True)
    buffer: ReplayBuffer = struct.field(pytree_node=True)
    log: DQNLog = struct.field(pytree_node=True)


class DQN(Agent):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    hparams: DQNHParams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        hparams: DQNHParams,
        optimiser: optax.GradientTransformation,
        backbone: nn.Module,
    ) -> DQN:
        critic = nn.Sequential(
            [
                backbone,
                nn.Dense(hparams.action_space.maximum),
            ]
        )
        return DQN(
            hparams=hparams,
            optimiser=optimiser,
            critic=critic,
        )

    def init(self, timestep: Timestep, *, key: KeyArray) -> DQNState:
        hparams = self.hparams
        assert isinstance(hparams.action_space, Discrete)
        iteration = jnp.asarray(0)
        params = self.critic.init(
            key, hparams.obs_space.sample(key)
        )
        params_target = jtu.tree_map(lambda x: x, params)  # copy params
        buffer = ReplayBuffer.create(
            timestep,
            hparams.replay_memory_size,
            hparams.n_steps,
        )
        opt_state = self.optimiser.init(params)
        return DQNState(
            iteration=iteration,
            params=params,
            params_target=params_target,
            opt_state=opt_state,
            buffer=buffer,
            log=DQNLog(),
        )

    def sample_action(
        self, train_state: DQNState, obs: Array, *, key: KeyArray, eval: bool = False
    ) -> Array:
        q_values = self.critic.apply(train_state.params, obs)
        eps = optax.polynomial_schedule(
            init_value=self.hparams.initial_exploration,
            end_value=self.hparams.final_exploration,
            transition_steps=self.hparams.final_exploration_frame
            - self.hparams.replay_start,
            transition_begin=self.hparams.replay_start,
            power=1.0,
        )(train_state.iteration)
        action = distrax.EpsilonGreedy(jnp.asarray(q_values), jnp.asarray(eps)).sample(seed=key)  # type: ignore
        return action

    def loss(
        self,
        params: Params,
        timesteps: Timestep,
        params_target: Params,
    ) -> Array:
        s_tm1 = timesteps.observation[:-1]
        s_t = timesteps.observation[1:]
        a_tm1 = timesteps.action[:-1][0]  # [0] because scalar
        r_t = timesteps.reward[:-1][0]  # [0] because scalar
        terminal_tm1 = timesteps.step_type[:-1] != TERMINATION
        discount_t = self.hparams.discount ** timesteps.t[:-1][0]  # [0] because scalar

        q_tm1 = jnp.asarray(self.critic.apply(params, s_tm1))
        q_t = jnp.asarray(self.critic.apply(params_target, s_t)) * terminal_tm1

        td_error = rlax.q_learning(
            q_tm1, a_tm1, r_t, discount_t, q_t, stop_target_gradients=True
        )
        td_error = jnp.asarray(td_error)
        td_loss = jnp.mean(0.5 * td_error**2)
        return td_loss

    def update(
        self,
        train_state: DQNState,
        transition: Timestep,
        *,
        key: KeyArray,
    ) -> DQNState:
        # update iteration
        iteration = jnp.asarray(train_state.iteration + 1, dtype=jnp.int32)

        # update memory
        buffer = train_state.buffer.add(transition)

        # update critic
        def _sgd_step(params, opt_state):
            transitions = buffer.sample(key, self.hparams.batch_size)
            loss_fn = lambda params, trans: jnp.mean(
                jax.vmap(self.loss, in_axes=(None, 0, None))(
                    params, trans, train_state.params_target
                )
            )
            loss, grads = jax.value_and_grad(loss_fn)(params, transitions)
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        cond = buffer.size() < self.hparams.replay_memory_size
        cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
        params, opt_state, loss = jax.lax.cond(
            cond,
            lambda p, o: _sgd_step(p, o),
            lambda p, o: (p, o, jnp.asarray(float("inf"))),
            train_state.params,
            train_state.opt_state,
        )

        # update target critic
        params_target = optax.periodic_update(
            params,
            train_state.params_target,
            jnp.asarray(iteration, dtype=jnp.int32),
            self.hparams.target_network_update_frequency,
        )

        # log
        log = DQNLog(
            iteration=iteration,
            critic_loss=loss,
            step_type=transition.step_type[-1],
            returns=train_state.log.returns
            + jnp.sum(
                self.hparams.discount ** transition.t[:-1] * transition.reward[:-1]
            ),
            buffer_size=buffer.size(),
        )

        # update train_state
        train_state = train_state.replace(
            iteration=iteration,
            opt_state=opt_state,
            params=params,
            params_target=params_target,
            buffer=buffer,
            log=log,
        )
        return train_state
