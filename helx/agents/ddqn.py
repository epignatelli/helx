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

from flax.core.scope import VariableDict as Params
from jax import Array
import jax.numpy as jnp
import optax

from ..mdp import Timestep
from .dqn import DQNHParams, DQNLog, DQNState, DQN
from ..mdp import TERMINATION


class DDQNHParams(DQNHParams):
    ...


class DDQNLog(DQNLog):
    ...


class DDQNState(DQNState):
    ...


class DDQN(DQN):
    """Implements a Double Deep Q-Network:
    Deep Reinforcement Learning with Double Q-learning,
    Arthur Guez, David Silver, Hado van Hasselt
    https://ojs.aaai.org/index.php/AAAI/article/view/10295"""

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

        a_t = jnp.argmax(jnp.asarray(self.critic.apply(params, s_t)) * terminal_tm1)
        q_t = self.critic.apply(params_target, s_t)
        q_target = r_t + discount_t * q_t[a_t]

        td_loss = optax.l2_loss(q_tm1[a_tm1] - q_target)
        return jnp.asarray(td_loss)
