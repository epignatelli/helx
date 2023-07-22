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

import dm_env
import dm_env.specs
import jax
import jax.numpy as jnp

from ..mdp import Action, Timestep
from ..spaces import Space, Discrete, Continuous
from .environment import EnvironmentWrapper


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
        return Continuous(shape=dm_space.shape, minimum=dm_space.minimum.min().item(), maximum=dm_space.maximum.max().item())
    else:
        raise NotImplementedError(
            "Cannot convert dm_env space of type {}".format(type(dm_space))
        )


class DmEnvWrapper(EnvironmentWrapper):
    """Static class to convert between dm_env and helx environments."""
    @classmethod
    def init(cls, env: dm_env.Environment) -> Tuple[DmEnvWrapper, Timestep]:
        self = cls(
            env=env,
            observation_space=to_helx(env.observation_spec()),
            action_space=to_helx(env.action_spec()),
            reward_space=to_helx(env.reward_spec()),
        )
        return self, self.reset(seed=0)

    def reset(self, seed: int | None = None) -> Timestep:
        next_step = self.env.reset()
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_dm_env(next_step)

    def step(self, action: Action) -> Timestep:
        next_step = self.env.step(action.item())
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_dm_env(next_step)

    def seed(self, seed: int) -> None:
        self._seed = seed
        self._key = jax.random.PRNGKey(self._seed)
        return
