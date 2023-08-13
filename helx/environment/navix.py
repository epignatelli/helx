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

import jax
import jax.numpy as jnp
import navix as nx
from jax.random import KeyArray

from ..spaces import Continuous, Discrete, Space
from ..mdp import TRANSITION, Timestep
from .environment import EnvironmentWrapper


@overload
def to_helx(space: nx.spaces.Discrete) -> Discrete:
    ...


@overload
def to_helx(space: nx.spaces.Continuous) -> Continuous:
    ...


@overload
def to_helx(space: nx.spaces.Space) -> Space:
    ...


def to_helx(space: nx.spaces.Space) -> Space:
    if isinstance(space, nx.spaces.Discrete):
        return Discrete(space.maximum.item(), shape=space.shape, dtype=space.dtype)
    elif isinstance(space, nx.spaces.Continuous):
        return Continuous(
            shape=space.shape,
            minimum=space.minimum.item(),
            maximum=space.maximum.item(),
        )
    else:
        raise NotImplementedError(
            "Cannot convert dm_env space of type {}".format(type(space))
        )


class NavixWrapper(EnvironmentWrapper):
    """Static class to convert between Gymnax environments and helx environments."""
    env: nx.environments.Environment

    @classmethod
    def wraps(cls, env: nx.environments.Environment) -> NavixWrapper:
        return cls(
            env=env,
            observation_space=to_helx(env.observation_space),
            action_space=to_helx(env.action_space),
            reward_space=Continuous(),
        )

    def reset(self, key: KeyArray) -> Timestep:
        timestep = self.env.reset(key)
        return Timestep(
            t=jnp.asarray(0),
            observation=timestep.observation,
            reward=timestep.reward,
            step_type=TRANSITION,
            action=jnp.asarray(-1),
            state=timestep.state,
            info=timestep.info
        )

    def _step(self, key: KeyArray, timestep: Timestep, action: jax.Array) -> Timestep:
        current_step = nx.environments.environment.Timestep(
            t=timestep.t,
            observation=timestep.observation,
            reward=timestep.reward,
            step_type=timestep.step_type,
            action=action,
            state=timestep.state,
            info=timestep.info
        )
        nexst_step = self.env.step(current_step, action)
        return Timestep(
            t=nexst_step.t,
            observation=nexst_step.observation,
            reward=nexst_step.reward,
            step_type=nexst_step.step_type,
            action=action,
            state=nexst_step.state,
            info=nexst_step.info
        )

