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

import dm_env
import dm_env.specs
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.random import KeyArray

from ..mdp import Action, Timestep, StepType
from ..spaces import Space, Discrete, Continuous
from .base import Environment


def discrete_from_dm_env(dm_space: dm_env.specs.DiscreteArray) -> Discrete:
    return Discrete.create(dm_space.num_values)


def continuous_from_dm_env(dm_space: dm_env.specs.BoundedArray) -> Continuous:
    shape = dm_space.shape
    minimum = jnp.asarray(dm_space.minimum)
    maximum = jnp.asarray(dm_space.maximum)
    return Continuous(shape=shape, dtype=dm_space.dtype, lower=minimum, upper=maximum)


def space_from_dm_env(dm_space: dm_env.specs.Array) -> Space:
    if isinstance(dm_space, dm_env.specs.DiscreteArray):
        return discrete_from_dm_env(dm_space)
    elif isinstance(dm_space, dm_env.specs.BoundedArray):
        return continuous_from_dm_env(dm_space)

    raise NotImplementedError(
        "Cannot convert dm_env space of type {}".format(type(dm_space))
    )


def timestep_from_dm_env(
    dm_step: dm_env.TimeStep, action: Action = -1, t: ArrayLike = 0, gamma: float  = 1.0
) -> Timestep:
    step_type = dm_step.step_type
    obs = jnp.asarray(dm_step.observation)
    reward = jnp.asarray(dm_step.reward)
    discount = dm_step.discount

    if dm_step.step_type == dm_env.StepType.LAST:
        step_type = StepType.TERMINATION
    elif discount is not None and float(discount) == 0.0:
        step_type = StepType.TRUNCATION
    else:
        step_type = StepType.TRANSITION

    requires_reset = step_type == StepType.TERMINATION
    return Timestep(
        observation=obs,
        reward=reward,
        step_type=step_type,
        action=action,
        t=t,
    )


class DmEnvAdapter(Environment[dm_env.Environment]):
    """Static class to convert between dm_env and helx environments."""

    @classmethod
    def create(cls, env: dm_env.Environment):
        return cls(
            env=env,
            # TODO(epignatelli): remove `type: ignore` when bsuite correctly typed
            observation_space=space_from_dm_env(env.observation_spec()),  # type: ignore
            # TODO(epignatelli): remove `type: ignore` when bsuite correctly typed
            action_space=space_from_dm_env(env.action_spec()),  # type: ignore
            reward_space=space_from_dm_env(env.reward_spec()),
        )

    def reset(self, key: KeyArray = 0) -> Timestep:
        next_step = self.env.reset()
        return timestep_from_dm_env(next_step)

    def step(self, current_state: Timestep, action: Action, key: KeyArray) -> Timestep:
        if current_state.is_terminal():
            return self.reset(key)

        next_step = self.env.step(action)
        return timestep_from_dm_env(next_step, action, t=current_state.t + 1)
