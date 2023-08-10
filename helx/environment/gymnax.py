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

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from gymnax.environments.environment import Environment as GymnaxEnvironment, EnvParams
from gymnax.environments.environment import EnvParams
from gymnax.environments.spaces import Space as GymnaxSpace, gymnax_space_to_gym_space
from jax.random import KeyArray

from ..spaces import Space, Continuous
from ..mdp import Timestep, TRANSITION, TERMINATION, TRUNCATION
from .environment import EnvironmentWrapper
from .gym import to_helx as gym_to_helx


def to_helx(gym_space: GymnaxSpace) -> Space:
    gym_space = gymnax_space_to_gym_space(gym_space)  # type: ignore
    return gym_to_helx(gym_space)  # type: ignore


def timestep_from_gym(
    obs, state, reward, done, info, action: Array, t: Array
) -> Timestep:
    return Timestep(
        observation=jnp.asarray(obs),
        reward=jnp.asarray(reward),
        step_type=(TRANSITION, TERMINATION)[done],
        action=jnp.asarray(action),
        t=t,
        state=state,
        info=info,
    )


class GymnaxWrapper(EnvironmentWrapper):
    """Static class to convert between Gymnax environments and helx environments."""
    env: GymnaxEnvironment
    params: EnvParams

    @classmethod
    def wraps(cls, env: Tuple[GymnaxEnvironment, EnvParams]) -> GymnaxWrapper:
        env_, params = env
        return cls(
            env=env_,
            observation_space=to_helx(env_.observation_space(params)),
            action_space=to_helx(env_.action_space(params)),
            reward_space=Continuous(),
            params=params,
        )

    def reset(self, key: KeyArray) -> Timestep:
        obs, state = self.env.reset(key, self.params)
        return Timestep(
            t=jnp.asarray(0),
            observation=jnp.asarray(obs),
            reward=jnp.asarray(0.0),
            step_type=TRANSITION,
            action=jnp.asarray(-1),
            state=state,
        )

    def _step(self, key: KeyArray, timestep: Timestep, action: jax.Array) -> Timestep:
        obs, state, reward, done, info = self.env.step(
            key=key, state=timestep.state, action=action, params=self.params
        )
        idx = 2 * done + jnp.asarray(
            timestep.t > self.params.max_steps_in_episode, dtype=jnp.int32
        )  # out-of-bounds returns clamps to 2
        step_type = jax.lax.switch(
            idx,
            (
                lambda: TRANSITION,
                lambda: TRUNCATION,
                lambda: TERMINATION,
            ),
        )
        return Timestep(
            t=timestep.t + 1,
            observation=jnp.asarray(obs),
            reward=jnp.asarray(reward),
            step_type=step_type,
            action=jnp.asarray(action),
            state=state,
        )
