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

import re
from collections import defaultdict
from typing import Dict, List

import gym
import gym.core
import gym.utils.seeding
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from gym.envs.registration import parse_env_id, registry
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import ImgObsWrapper
from gym.envs.registration import EnvSpec

from ..logging import get_logger
from ..mdp import Action, StepType, Timestep
from ..spaces import Continuous, Space
from .base import Environment

logging = get_logger()


class FromGymEnv(Environment[gym.Env]):
    """Static class to convert between gym and helx environments."""

    def __init__(self, env: gym.core.Env):
        if isinstance(env.unwrapped, MiniGridEnv):
            msg = (
                "String arrays are not supported by helx yet."
                " The `mission` field of the observations returned by"
                " MiniGrid environments contain string arrays."
                " We get rid of the `mission` field by wrapping `env`"
                " around an `ImgObsWrapper`."
            )
            logging.warning(msg)
            env = ImgObsWrapper(env)

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
        try:
            obs, _ = self._env.reset(seed=seed)
        except TypeError:
            # TODO(epignatelli): remove try/except when gym3 is updated.
            # see: https://github.com/openai/gym3/issues/8
            obs, _ = self._env.reset()
        self._current_observation = jnp.asarray(obs)
        return Timestep(obs, None, StepType.TRANSITION)

    def step(self, action: Action) -> Timestep:
        action_ = np.asarray(action)
        next_step = self._env.step(action_)
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_gym(next_step)

    def seed(self, seed: int) -> None:
        self._env.np_random, seed = gym.utils.seeding.np_random(seed)
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human"):
        self._env.render_mode = mode
        return self._env.render()

    def close(self) -> None:
        return self._env.close()

    def name(self) -> str:
        return self._env.unwrapped.__class__.__name__


def list_envs(namespace: str) -> List[str]:
    env_specs: Dict[str, EnvSpec] = {
        k: v for k, v in registry.items() if namespace.lower() in v.entry_point.lower()
    }
    return list(env_specs.keys())
