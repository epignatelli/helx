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
import jax
import jax.numpy as jnp
from chex import Array

from ..mdp import Action, StepType, Timestep
from ..spaces import Space, Continuous
from .base import Environment


class BraxAdapter(Environment[brax.envs.Env]):
    """Static class to convert between bsuite.Environment and helx environments."""

    def __init__(self, env: brax.envs.Env):
        super().__init__(env)
        self._current_state = None

    def action_space(self) -> Space:
        if self._action_space is None:
            self._action_space = Continuous((self._env.action_size, ))
        return self._action_space

    def observation_space(self) -> Space:
        if self._observation_space is None:
            self._observation_space = Continuous((self._env.observation_size,))
        return self._observation_space

    def reward_space(self) -> Space:
        if self._reward_space is None:
            self._reward_space = Continuous(())

        return self._reward_space

    def state(self) -> Array:
        if hasattr(self._env, "_get_observation"):
            return self._env._get_observation()  # type: ignore

        if self._current_observation is None:
            raise ValueError(
                "Environment not initialized. Run `reset` first to produce a starting state."
            )
        return self._current_observation

    def reset(self, seed: int | None = None) -> Timestep:
        key = jax.random.PRNGKey(seed) if seed is not None else self._key
        next_step = self._env.reset(key)  # type: ignore
        observation = next_step.obs
        reward = next_step.reward
        step_type = StepType(next_step.done)  # TODO(epignatelli): this will break with vector env

        self._current_state = next_step
        self._current_observation = jnp.asarray(observation)
        return Timestep(observation, reward, step_type)

    def step(self, action: Action) -> Timestep:
        next_step = self._env.step(self._current_state, action)  # type: ignore
        observation = next_step.obs
        reward = next_step.reward
        step_type = StepType(next_step.done)  # TODO(epignatelli): this will break with vector env

        self._current_state = next_step
        self._current_observation = observation
        return Timestep(observation, reward, step_type)

    def seed(self, seed: int) -> None:
        self._seed = seed
        self._key = jax.random.PRNGKey(self._seed)
        return

    def render(self, mode: str = "human"):
        # TODO: Handle mode
        current_state = self.state()
        return current_state

    def close(self) -> None:
        return
