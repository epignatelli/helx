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

import gymnasium
import gymnasium.spaces
import gymnasium.utils.seeding
import jax.numpy as jnp
import numpy as np
from gymnasium.utils.step_api_compatibility import (
    TerminatedTruncatedStepType as GymnasiumTimestep,
)
from jax.typing import ArrayLike

from ..mdp import Action, StepType, Timestep
from ..spaces import Continuous, Discrete, Space
from .base import Environment


def continuous_from_gymnasium(gym_space: gymnasium.spaces.Box) -> Continuous:
    shape = gym_space.shape
    minimum = jnp.asarray(gym_space.low)
    maximum = jnp.asarray(gym_space.high)
    return Continuous(shape=shape, dtype=gym_space.dtype, lower=minimum, upper=maximum)


def discrete_from_gymnasium(gym_space: gymnasium.spaces.Discrete) -> Discrete:
    return Discrete.create(gym_space.n)


def space_from_gymnasium(gym_space: gymnasium.spaces.Space) -> Space:
    if isinstance(gym_space, gymnasium.spaces.Discrete):
        return discrete_from_gymnasium(gym_space)
    elif isinstance(gym_space, gymnasium.spaces.Box):
        return continuous_from_gymnasium(gym_space)
    else:
        raise NotImplementedError(
            "Cannot convert gym space of type {}".format(type(gym_space))
        )


def timestep_from_gymnasium(
    gym_step: GymnasiumTimestep, action: Action = -1, t: ArrayLike = 0
) -> Timestep:
    obs, reward, terminated, truncated, _ = gym_step

    if terminated:
        step_type = StepType.TERMINATION
    elif truncated:
        step_type = StepType.TRUNCATION
    else:
        step_type = StepType.TRANSITION

    obs = jnp.asarray(obs)
    reward = jnp.asarray(reward)
    action = jnp.asarray(action)
    return Timestep(
        observation=obs,
        reward=reward,
        step_type=step_type,
        action=action,
        t=t,
    )


class GymnasiumAdapter(Environment[gymnasium.Env]):
    """Static class to convert between gymnasium and helx environments."""

    @classmethod
    def create(cls, env: gymnasium.Env):
        return cls(
            env=env,
            action_space=space_from_gymnasium(env.action_space),
            observation_space=space_from_gymnasium(env.observation_space),
            reward_space=Continuous(
                (), lower=env.reward_range[0], upper=env.reward_range[1]
            ),
        )

    def reset(self, seed: int | None = None) -> Timestep:
        obs, _ = self.env.reset(seed=seed)
        return Timestep(
            observation=obs, reward=0, step_type=StepType.TRANSITION, action=-1, t=0
        )

    def step(self, current_timestep: Timestep, action: Action) -> Timestep:
        if current_timestep.is_terminal():
            return current_timestep

        action_ = np.asarray(action)
        next_step = self.env.step(action_)
        return timestep_from_gymnasium(
            next_step, action=action, t=current_timestep.t + 1
        )
