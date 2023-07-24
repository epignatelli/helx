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


"""A set of functions to interoperate between the most common
RL environment interfaces, like `gym`, `gymnasium`, `dm_env`, `bsuite and others."""
from __future__ import annotations

import abc
from typing import Any, Tuple

from flax import struct
import jax
from jax.random import KeyArray

from ..mdp import Action, StepType, Timestep
from ..spaces import Space


class Environment(struct.PyTreeNode):
    action_space: Space = struct.field(pytree_node=False)
    observation_space: Space = struct.field(pytree_node=False)
    reward_space: Space = struct.field(pytree_node=False)

    @abc.abstractmethod
    def reset(self, key: KeyArray) -> Timestep:
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, key: KeyArray, timestep: Timestep, action: Action) -> Timestep:
        raise NotImplementedError()

    def name(self) -> str:
        return self.__class__.__name__


class EnvironmentWrapper(Environment):
    env: Any = struct.field(pytree_node=False)
    observation_space: Space = struct.field(pytree_node=False)
    action_space: Space = struct.field(pytree_node=False)
    reward_space: Space = struct.field(pytree_node=False)

    @abc.abstractclassmethod
    def to_helx(self, env: Any) -> EnvironmentWrapper:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, key: KeyArray | int) -> Timestep:
        raise NotImplementedError()

    @abc.abstractmethod
    def _step(
        self, key: KeyArray | int, timestep: Timestep, action: Action
    ) -> Timestep:
        raise NotImplementedError()

    def step(self, key: KeyArray | int, timestep: Timestep, action: Action) -> Timestep:
        # autoreset
        next_timestep = jax.lax.cond(
            timestep.step_type == StepType.TRANSITION,
            lambda timestep: self._step(key, timestep, action),
            lambda timestep: self.reset(key),
            timestep,
        )
        return next_timestep

    def unwrapped(self) -> Any:
        return self.env

    def name(self) -> str:
        env = self.env
        while hasattr(env, "unwrapped"):
            unwrapped = getattr(env, "unwrapped")
            if unwrapped == env:
                break
            env = unwrapped
        return env.__class__.__name__
