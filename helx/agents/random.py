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

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from chex import Array

from .agent import Agent, HParams
from ..mdp import Timestep


class Random(Agent):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    def __init__(
        self,
        hparams: HParams,
        seed: int,
    ):
        self.key: jax.random.KeyArray = jax.random.PRNGKey(seed)
        self.hparams: HParams = hparams
        self.iteration: int = 0

    def sample_action(self, observation: Array, eval: bool = False, **kwargs) -> Array:
        return self.hparams.action_space.sample(self.key)

    def update(self, episode: Timestep) -> Array:
        return jnp.asarray(())
