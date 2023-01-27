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
from typing import Any, Generic, TypeVar

import jax
from chex import Array

from jax.random import KeyArray

from ..mdp import Action, Timestep
from ..spaces import Space

T = TypeVar("T")


class Environment(abc.ABC, Generic[T]):
    def __init__(self, env: Any):
        self._action_space: Space | None = None
        self._observation_space: Space | None = None
        self._reward_space: Space | None = None
        self._current_observation: Array | None = None
        self._seed: int = 0
        self._key: KeyArray = jax.random.PRNGKey(self._seed)
        self._env: T = env

    @abc.abstractmethod
    def action_space(self) -> Space:
        raise NotImplementedError()

    @abc.abstractmethod
    def observation_space(self) -> Space:
        raise NotImplementedError()

    @abc.abstractmethod
    def reward_space(self) -> Space:
        raise NotImplementedError()

    @abc.abstractmethod
    def state(self) -> Array:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> Timestep:
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, action: Action) -> Timestep:
        raise NotImplementedError()

    @abc.abstractmethod
    def seed(self, seed: int) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self, mode: str = "human"):
        raise NotImplementedError()

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    def name(self) -> str:
        return self._env.__class__.__name__

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __del__(self):
        self.close()
