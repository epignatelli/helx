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

import brax.envs
import jax.numpy as jnp
from jax.random import KeyArray

from ..mdp import Action, StepType, Timestep
from ..spaces import Continuous
from .base import Environment


class BraxAdapter(Environment[brax.envs.Env]):
    """Static class to convert between bsuite.Environment and helx environments."""

    @classmethod
    def create(cls, env: brax.envs.Env):
        return cls(
            observation_space=Continuous((env.observation_size,)),
            action_space=Continuous((env.action_size,)),
            reward_space=Continuous(()),
            env=env,
        )

    def reset(self, key: KeyArray) -> Timestep:
        # TODO(epignatelli): wrongly typed in brax/jax, KeyArray is not Array
        next_step = self.env.reset(key)  # type: ignore
        return Timestep(
            observation=next_step.obs,
            reward=next_step.reward,
            step_type=StepType(next_step.done),
            action=-1,
            t=0,
            info={"state": next_step},
        )

    def step(
        self, current_timestep: Timestep, action: Action, seed: int = 0
    ) -> Timestep:
        if current_timestep.is_terminal():
            return self.reset(seed)

        action = jnp.asarray(action)
        next_step = self.env.step(current_timestep.info["state"], action)
        return Timestep(
            observation=next_step.obs,
            reward=next_step.reward,
            step_type=StepType(next_step.done),
            action=action,
            t=current_timestep.t + 1,
            info={"state": next_step},
        )
