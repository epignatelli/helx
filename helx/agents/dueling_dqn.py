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

import jax.numpy as jnp
import optax
from flax import struct
import flax.linen as nn

from .dqn import DQNHParams, DQNLog, DQNState, DQN
from helx.base.modules import Split, Merge, Parallel


class DuelingDQNHParams(DQNHParams):
    ...


class DuelingDQNLog(DQNLog):
    ...


class DuelingDQNState(DQNState):
    ...


class DuelingDQN(DQN):
    """Dueling DQN agent as described in https://arxiv.org/abs/1511.06581
    Uses the average operator version to combine the advantage and value functions."""

    hparams: DuelingDQNHParams = struct.field(pytree_node=True)
    optimiser: optax.GradientTransformation = struct.field(pytree_node=True)
    critic: nn.Module = struct.field(pytree_node=True)

    @classmethod
    def create(
        cls,
        hparams: DuelingDQNHParams,
        optimiser: optax.GradientTransformation,
        backbone: nn.Module,
    ) -> DuelingDQN:
        critic = nn.Sequential(
            [
                backbone,
                Split(2),
                Parallel((nn.Dense(1), nn.Dense(hparams.action_space.maximum))),  # v, A
                Merge(
                    lambda inputs: inputs[0]
                    + (inputs[1] - jnp.mean(inputs[1], axis=-1))
                ),  # q = v + (A - mean(A))
            ]
        )
        return DuelingDQN(
            hparams=hparams,
            optimiser=optimiser,
            critic=critic,
        )
