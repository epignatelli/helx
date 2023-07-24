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

from typing import Tuple, overload

import gym
import gym.core
import gym.spaces
import gym.utils.seeding
import jax.numpy as jnp
import numpy as np

from ..mdp import Action, StepType, Timestep
from ..spaces import Continuous, Discrete, Space
from .environment import EnvironmentWrapper


@overload
def to_helx(gym_space: gym.spaces.Discrete) -> Discrete:
    ...


@overload
def to_helx(gym_space: gym.spaces.Box) -> Continuous:
    ...


def to_helx(gym_space: gym.spaces.Space) -> Space:
    if isinstance(gym_space, gym.spaces.Discrete):
        return Discrete(gym_space.n)
    elif isinstance(gym_space, gym.spaces.Box):
        return Continuous(
            shape=gym_space.shape,
            minimum=gym_space.low.min().item(),
            maximum=gym_space.high.max().item(),
        )
    else:
        raise NotImplementedError(
            "Cannot convert dm_env space of type {}".format(type(gym_space))
        )


class GymWrapper(EnvironmentWrapper):
    """Static class to convert between gym and helx environments."""

    @classmethod
    def to_helx(cls, env: gym.Env) -> GymWrapper:
        self = cls(
            env=env,
            observation_space=to_helx(env.observation_space),  # type: ignore
            action_space=to_helx(env.action_space),  # type: ignore
            reward_space=to_helx(env.reward_range),  # type: ignore
        )
        return self

    def reset(self, seed: int | None = None) -> Timestep:
        try:
            obs, _ = self.env.reset(seed=seed)
        except TypeError:
            # TODO(epignatelli): remove try/except when gym3 is updated.
            # see: https://github.com/openai/gym3/issues/8
            obs, _ = self.env.reset()
        self._current_observation = jnp.asarray(obs)
        return Timestep(obs, None, StepType.TRANSITION)

    def step(self, action: Action) -> Timestep:
        action_ = np.asarray(action)
        next_step = self.env.step(action_)
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_gym(next_step)
