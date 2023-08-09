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

from flax.core.scope import VariableDict as Params
from jax import Array

from ..mdp import Timestep
from .dqn import DQNHParams, DQNLog, DQNState, DQN
from .. import losses


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
        transition: Timestep,
        params_target: Params,
    ) -> Array:
        return losses.double_dqn_loss(
            transition,
            self.critic,
            params,
            params_target,
            self.hparams.discount,
        )
