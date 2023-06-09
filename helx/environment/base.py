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
from functools import wraps
from typing import Any, TypeVar

from flax import struct
from jax.random import KeyArray

from ..mdp import Action, Timestep
from ..spaces import Space

T = TypeVar("T")


class Environment(struct.PyTreeNode):
    env: Any = struct.field(pytree_node=False)
    observation_space: Space = struct.field(pytree_node=False)
    action_space: Space = struct.field(pytree_node=False)
    reward_space: Space = struct.field(pytree_node=False)

    @abc.abstractclassmethod
    def _create(self, env: Any) -> Environment:
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset(self, key: KeyArray) -> Timestep:
        raise NotImplementedError()

    @abc.abstractmethod
    def _step(
        self, current_timestep: Timestep, action: Action, key: KeyArray
    ) -> Timestep:
        raise NotImplementedError()

    def reset(self, key: KeyArray) -> Timestep:
        next_timestep = self._reset(key)
        return next_timestep

    def step(
        self, current_timestep: Timestep, action: Action, key: KeyArray
    ) -> Timestep:
        next_timestep = self._step(current_timestep, action, key)
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
