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

import dm_env
import dm_env.specs
import jax
import jax.numpy as jnp
from jax.random import KeyArray

from helx.base.mdp import Timestep, TERMINATION, TRANSITION, TRUNCATION
from helx.base.spaces import Space, Discrete, Continuous
from .environment import EnvironmentWrapper


@overload
def to_helx(dm_space: dm_env.specs.DiscreteArray) -> Discrete:
    ...


@overload
def to_helx(dm_space: dm_env.specs.BoundedArray) -> Continuous:
    ...


@overload
def to_helx(dm_space: dm_env.specs.Array) -> Continuous:
    ...


def to_helx(dm_space: dm_env.specs.Array) -> Space:
    if isinstance(dm_space, dm_env.specs.DiscreteArray):
        return Discrete(dm_space.num_values)
    elif isinstance(dm_space, dm_env.specs.BoundedArray):
        return Continuous(
            shape=dm_space.shape,
            minimum=dm_space.minimum.min().item(),
            maximum=dm_space.maximum.max().item(),
        )
    elif isinstance(dm_space, dm_env.specs.Array):
        return Continuous(shape=dm_space.shape)
    else:
        raise NotImplementedError(
            "Cannot convert dm_env space of type {}".format(type(dm_space))
        )


def timestep_to_helx(
    timestep: dm_env.TimeStep, action: jax.Array, t: jax.Array
) -> Timestep:
    step_type = timestep.step_type
    obs = jnp.asarray(timestep.observation)
    reward = jnp.asarray(timestep.reward)
    discount = timestep.discount

    if timestep.step_type == dm_env.StepType.LAST:
        step_type = TERMINATION
    elif discount is not None and float(discount) == 0.0:
        step_type = TRUNCATION
    else:
        step_type = TRANSITION

    return Timestep(
        observation=obs,
        reward=reward,
        step_type=step_type,
        action=action,
        t=t,
        state=None,
    )


class DmEnvWrapper(EnvironmentWrapper):
    """Static class to convert between dm_env and helx environments."""
    env: dm_env.Environment

    @classmethod
    def wraps(cls, env: dm_env.Environment) -> DmEnvWrapper:
        self = cls(
            env=env,
            observation_space=to_helx(env.observation_spec()),  # type: ignore
            action_space=to_helx(env.action_spec()),  # type: ignore
            reward_space=to_helx(env.reward_spec()),  # type: ignore
        )
        return self

    def reset(self, key: KeyArray | int) -> Timestep:
        next_step = self.env.reset()
        return timestep_to_helx(next_step, jnp.asarray(-1), jnp.asarray(0))

    def _step(self, key: KeyArray, timestep: Timestep, action: jax.Array) -> Timestep:
        next_step = self.env.step(action.item())
        return timestep_to_helx(next_step, action, timestep.t + 1)
