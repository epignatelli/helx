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

from typing import Any, Tuple
import distrax
import jax

import jax.numpy as jnp
import rlax
from chex import Array
from flax import linen as nn
from flax import struct
from jax.random import KeyArray
import optax
from optax import GradientTransformation, OptState

from ..mdp import StepType, Timestep
from ..spaces import Discrete

from ..memory import ReplayBuffer
from .agent import Agent, HParams, Log


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
    buffer_size: Array
    critic_loss: Array


class DQN(Agent):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    # Static state
    hparams: DQNHParams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)
    # Dynamic state
    iteration: Array = struct.field(pytree_node=True)
    params_critic: nn.FrozenDict = struct.field(pytree_node=True)
    params_target: nn.FrozenDict = struct.field(pytree_node=True)
    opt_state: OptState = struct.field(pytree_node=True)
    buffer: ReplayBuffer = struct.field(pytree_node=True)

    @classmethod
    def create(
        cls,
        key: KeyArray,
        hparams: DQNHParams,
        optimiser: GradientTransformation,
        backbone: nn.Module,
    ):
        # config
        assert isinstance(hparams.action_space, Discrete)
        iteration = 0
        critic = nn.Sequential([backbone, nn.Dense(int(hparams.action_space.maximum))])
        params = critic.init(
            key, hparams.obs_space.sample(key), hparams.action_space.sample(key)
        )
        params_target = params_critic.copy({})  # type: ignore
        example_item = (
            hparams.obs_space.sample(key),
            hparams.action_space.sample(key),
            0.0,
            hparams.obs_space.sample(key),
            False,
        )
        buffer = ReplayBuffer.create(example_item, hparams.replay_memory_size)
        opt_state = optimiser.init(params)

        agent = cls(
            hparams=hparams,
            optimiser=optimiser,
            critic=critic,
            params=params,  # type: ignore
            params_target=params_target,
            opt_state=opt_state,
            iteration=iteration,
            buffer=buffer,
        )

        return agent

    def sample_action(self, obs: Array, key: KeyArray, eval: bool = False) -> Array:
        q_values = self.critic.apply(self.params, obs)
        eps = optax.polynomial_schedule(
            init_value=self.hparams.initial_exploration,
            end_value=self.hparams.final_exploration,
            transition_steps=self.hparams.final_exploration_frame
            - self.hparams.replay_start,
            transition_begin=self.hparams.replay_start,
            power=1.0,
        )(self.iteration)
        action = distrax.EpsilonGreedy(q_values, eps).sample(seed=key)  # type: ignore
        return action

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Timestep,
    ) -> Tuple[Array, Any]:
        s_tm1 = transition.observation[0:-1]
        s_t = transition.observation[1:]
        a_tm1 = transition.action
        r_t = transition.reward
        d_tm1 = transition.step_type == transition.step_type.TERMINATION
        discount_t = self.hparams.discount ** transition.t[:-1]

        q_tm1 = jnp.asarray(self.critic.apply(params, s_tm1))
        q_t = jnp.asarray(self.critic.apply(self.params_target, s_t)) * (
            d_tm1 == StepType.TERMINATION
        )

        td_error = rlax.q_learning(
            q_tm1, a_tm1, r_t, discount_t, q_t, stop_target_gradients=True
        )
        loss = jnp.mean(td_error**2 / 2)
        return loss, ()

    def update(
        self, transition: Timestep, key: KeyArray, cached_log: DQNLog
    ) -> Tuple[Agent, DQNLog]:
        # update iteration
        iteration = jnp.asarray(self.iteration + 1, dtype=jnp.int32)

        # update memory
        buffer = self.buffer.add(transition)

        # update critic
        def _sgd_step(params, opt_state):
            transitions = buffer.sample(key, self.hparams.batch_size)
            loss_fn = lambda params, trans: jnp.mean(
                jax.vmap(self.loss, in_axes=(0, None)(params, trans))  # type: ignore
            )
            loss, grads = jax.value_and_grad(loss_fn)(params, transitions)
            opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, opt_state)
            return params, opt_state, loss

        cond = buffer.size() < self.hparams.replay_memory_size
        cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
        params, opt_state, loss = jax.lax.cond(
            cond,
            lambda p, o: _sgd_step(p, o),
            lambda p, o: (p, o, jnp.asarray([])),
            self.params,
            self.opt_state,
        )

        # update target critic
        params_target = optax.periodic_update(
            self.params_target,
            self.critic_target.parameters,  # type: ignore
            jnp.asarray(iteration, dtype=jnp.int32),
            self.hparams.target_network_update_frequency,
        )

        # log
        log = DQNLog(
            iteration=iteration,
            loss=loss,
            step_type=transition.step_type,
            returns=cached_log.returns
            + self.hparams.discount ** transition.t[:-1] * transition.reward,
            buffer_size=buffer.size(),
            critic_loss=loss,
        )

        # update agent
        updated = self.replace(
            iteration=iteration,
            opt_state=opt_state,
            params=params,
            params_target=params_target,
            buffer=buffer,
        )
        return updated, log
