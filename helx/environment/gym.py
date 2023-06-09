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

import logging
from typing import Any, Dict, List

import gym
import gym.core
import gym.spaces
import gym.utils.seeding
import jax.numpy as jnp
from jax.random import KeyArray
import numpy as np
from gym.envs.registration import EnvSpec, registry
from gym.utils.step_api_compatibility import TerminatedTruncatedStepType as GymTimestep
from jax.typing import ArrayLike
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper

from ..mdp import Action, StepType, Timestep
from ..spaces import Continuous, Discrete, Space
from .base import Environment


def continuous_from_gym(gym_space: gym.spaces.Box) -> Continuous:
    shape = gym_space.shape
    minimum = jnp.asarray(gym_space.low)
    maximum = jnp.asarray(gym_space.high)
    return Continuous(shape=shape, dtype=gym_space.dtype, lower=minimum, upper=maximum)


def discrete_from_gym(gym_space: gym.spaces.Discrete) -> Discrete:
    return Discrete.create(gym_space.n)


def space_from_gym(gym_space: gym.spaces.Space) -> Space:
    if isinstance(gym_space, gym.spaces.Discrete):
        return discrete_from_gym(gym_space)
    elif isinstance(gym_space, gym.spaces.Box):
        return continuous_from_gym(gym_space)
    else:
        raise NotImplementedError(
            "Cannot convert gym space of type {}".format(type(gym_space))
        )


def timestep_from_gym(
    gym_step: GymTimestep, action: Action = -1, t: ArrayLike = 0
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


class GymAdapter(Environment[gym.Env]):
    """Static class to convert between gym and helx environments."""

    @classmethod
    def create(cls, env: gym.core.Env):
        if isinstance(env.unwrapped, MiniGridEnv):
            msg = (
                "String arrays are not supported by helx yet."
                " The `mission` field of the observations returned by"
                " MiniGrid environments contain string arrays."
                " We get rid of the `mission` field by wrapping `env`"
                " around an `ImgObsWrapper`."
            )
            logging.warning(msg)
            env = ImgObsWrapper(env)  # type: ignore

        return cls(
            env=env,
            observation_space=space_from_gym(env.observation_space),
            action_space=space_from_gym(env.action_space),
            reward_space=Continuous(
                (), lower=env.reward_range[0], upper=env.reward_range[1]
            ),
        )

    def reset(self, key: KeyArray) -> Timestep:
        try:
            obs, _ = self.env.reset(seed=int(key[0]))
        except TypeError:
            # TODO(epignatelli): remove try/except when gym3 is updated.
            # see: https://github.com/openai/gym3/issues/8
            obs, _ = self.env.reset()
        # step_type is unclear from reset. What if the first state is also the last (e.g., a bandit)?
        return Timestep(
            observation=obs,
            reward=0.0,
            step_type=StepType.TRANSITION,
            action=-1,
            t=0,
        )

    def step(
        self, current_timestep: Timestep, action: Action, key: KeyArray
    ) -> Timestep:
        if current_timestep.is_terminal():
            current_timestep = self.reset(key)

        next_step = self.env.step(action)
        return timestep_from_gym(next_step, action, current_timestep.t + 1)


def list_envs(namespace: str) -> List[str]:
    env_specs: Dict[str, EnvSpec] = {
        k: v for k, v in registry.items() if namespace.lower() in v.entry_point.lower()
    }
    return list(env_specs.keys())
