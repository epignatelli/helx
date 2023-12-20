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


import bsuite.environments
import dm_env.specs
from jax import Array
from jax.random import KeyArray
import jax.numpy as jnp

from helx.base.mdp import Timestep
from helx.base.spaces import Space, Discrete, Continuous
from .environment import EnvironmentWrapper
from .dm_env import timestep_to_helx


@overload
def to_helx(dm_space: dm_env.specs.DiscreteArray) -> Discrete:
    ...


@overload
def to_helx(dm_space: dm_env.specs.BoundedArray) -> Continuous:
    ...


def to_helx(dm_space: dm_env.specs.Array) -> Space:
    if isinstance(dm_space, dm_env.specs.DiscreteArray):
        return Discrete(dm_space.num_values)
    elif isinstance(dm_space, dm_env.specs.BoundedArray):
        return Continuous(
            shape=dm_space.shape,
            minimum=dm_space.minimum.item(),
            maximum=dm_space.maximum.item(),
        )
    else:
        raise NotImplementedError(
            "Cannot convert dm_env space of type {}".format(type(dm_space))
        )


class BsuiteWrapper(EnvironmentWrapper):
    """Static class to convert between bsuite.Environment and helx environments."""
    env: bsuite.environments.Environment

    @classmethod
    def wraps(cls, env: bsuite.environments.Environment) -> BsuiteWrapper:
        self = cls(
            env=env,
            observation_space=to_helx(env.observation_spec()),  # type: ignore
            action_space=to_helx(env.action_spec()),  # type: ignore
            reward_space=to_helx(env.reward_spec()),  # type: ignore
        )
        return self

    def reset(self, key: int | None = None) -> Timestep:
        next_step = self.env.reset()
        return timestep_to_helx(next_step, jnp.asarray(-1), jnp.asarray(0))

    def _step(self, key: KeyArray, timestep: Timestep, action: Array) -> Timestep:
        next_step = self.env.step(action.item())
        return timestep_to_helx(next_step, action, timestep.t + 1)
