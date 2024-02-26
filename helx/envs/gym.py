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

from typing import overload

import numpy as np
import gym
import gym.core
import gym.spaces
import gym.utils.seeding
import gym.vector
from gym.utils.step_api_compatibility import (
    TerminatedTruncatedStepType as GymTimestep,
    DoneStepType,
    convert_to_terminated_truncated_step_api,
)

from jax import Array
import jax.numpy as jnp
from jax.random import KeyArray
import jax.tree_util as jtu

from helx.base.mdp import Timestep, StepType
from helx.base.spaces import Continuous, Discrete, Space
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


def timestep_from_gym(
    gym_step: GymTimestep | DoneStepType, action: Array, t: Array
) -> Timestep:
    gym_step = convert_to_terminated_truncated_step_api(gym_step)
    obs, reward, terminated, truncated, _ = gym_step

    if terminated:
        step_type = StepType.TERMINATION
    elif truncated:
        step_type = StepType.TRUNCATION
    else:
        step_type = StepType.TRANSITION

    obs = jtu.tree_map(lambda x: jnp.asarray(x), obs)
    reward = jnp.asarray(reward)
    action = jnp.asarray(action)
    return Timestep(
        observation=obs,
        reward=reward,
        step_type=step_type,
        action=action,
        t=t,
        state=None,
    )


class GymWrapper(EnvironmentWrapper):
    """Static class to convert between gym and helx environments."""

    env: gym.Env

    @classmethod
    def wraps(cls, env: gym.Env) -> GymWrapper:
        if isinstance(env, gym.vector.AsyncVectorEnv):
            raise NotImplementedError(
                "AsyncVectorEnv is not yet supported with JAX. \
                Please use SyncVectorEnv instead."
            )
        self = cls(
            env=env,
            observation_space=to_helx(env.observation_space),  # type: ignore
            action_space=to_helx(env.action_space),  # type: ignore
            reward_space=to_helx(env.reward_range),  # type: ignore
        )
        return self

    def reset(self, seed: int | None = None) -> Timestep:
        if isinstance(self.env, gym.vector.VectorEnv):
            seed = [seed] * self.env.num_envs  # type: ignore
        try:
            timestep = self.env.reset(seed=seed)
        except TypeError:
            # TODO(epignatelli): remove try/except when gym3 is updated.
            # see: https://github.com/openai/gym3/issues/8
            timestep = self.env.reset()
        return timestep_from_gym(timestep, action=jnp.asarray(-1), t=jnp.asarray(0))  # type: ignore

    def _step(self, key: KeyArray, timestep: Timestep, action: Array) -> Timestep:
        next_step = self.env.step(np.asarray(action))
        return timestep_from_gym(next_step, action, timestep.t + 1)
