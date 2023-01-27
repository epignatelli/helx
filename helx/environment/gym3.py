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

from typing import cast

import gym3
import gym3.interop
import jax.numpy as jnp
from chex import Array
import numpy as np
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

from .base import Environment
from ..mdp import Action, StepType, Timestep
from ..spaces import Continuous, Space


class FromGym3Env(Environment[gym3.interop.ToGymEnv]):
    """Static class to convert between gym3 and helx environments.
    An example of gym3 environments is
    procgen: https://github.com/openai/procgen"""

    def __init__(self, env: gym3.interop.ToGymEnv):
        super().__init__(env)

    def action_space(self) -> Space:
        if self._action_space is not None:
            return self._action_space

        self._action_space = Space.from_gym(self._env.action_space)
        return self._action_space

    def observation_space(self) -> Space:
        if self._observation_space is not None:
            return self._observation_space

        self._observation_space = Space.from_gym(self._env.observation_space)
        return self._observation_space

    def reward_space(self) -> Space:
        if self._reward_space is not None:
            return self._reward_space

        minimum = self._env.reward_range[0]
        maximum = self._env.reward_range[1]
        self._reward_space = Continuous((1,), (minimum,), (maximum,))
        return self._reward_space

    def state(self) -> Array:
        if self._current_observation is None:
            raise ValueError(
                "Environment not initialized. Run `reset` first, to set a starting state."
            )
        return self._current_observation

    def reset(self, seed: int | None = None) -> Timestep:
        # TODO(epignatelli): remove try/except when gym3 is updated.
        # see: https://github.com/openai/gym3/issues/8
        obs = self._env.reset()
        self._current_observation = jnp.asarray(obs)
        return Timestep(obs, None, StepType.TRANSITION)

    def step(self, action: Action) -> Timestep:
        action_ = np.asarray(action)
        next_step = self._env.step(action_)
        next_step = convert_to_terminated_truncated_step_api(next_step)
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_gym(next_step)

    def seed(self, seed: int) -> None:
        self._env.seed(seed)  # type: ignore

    def render(self, mode: str = "human"):
        return self._env.render(mode)

    def close(self) -> None:
        return self._env.close()

    def name(self) -> str:
        return self._env.unwrapped.__class__.__name__
