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

import abc
from typing import Type

from flax import struct
import jax
import jax.numpy as jnp
from jax.random import KeyArray

from jax.core import Shape
from jax import Array
from jax.typing import ArrayLike


POS_INF = 1e16
NEG_INF = -1e16


class Space(abc.ABC, struct.PyTreeNode):
    shape: Shape
    dtype: Type = jnp.float32
    lower: ArrayLike = NEG_INF
    upper: ArrayLike = POS_INF

    @abc.abstractmethod
    def sample(self, key: KeyArray) -> Array:
        raise NotImplementedError()


class Discrete(Space):
    @classmethod
    def create(cls, n_actions: int, action_shape=()):
        return cls(shape=action_shape, dtype=jnp.int32, lower=0, upper=n_actions - 1)

    def sample(self, key: KeyArray) -> Array:
        return jax.random.randint(key, self.shape, self.lower, self.upper, self.dtype)


class Continuous(Space):
    @classmethod
    def create(
        cls,
        action_shape=(),
        dtype=jnp.float32,
        lower_bound=NEG_INF,
        upper_bound=POS_INF,
    ):
        return cls(
            shape=action_shape, dtype=jnp.int32, lower=lower_bound, upper=upper_bound
        )

    def sample(self, key: KeyArray) -> Array:
        lower = self.lower
        upper = self.upper
        if jnp.issubdtype(self.dtype, jnp.integer):
            return jax.random.randint(key, self.shape, lower, upper, dtype=self.dtype)
        elif jnp.issubdtype(self.dtype, jnp.floating):
            # TODO(epignatelli): jax bug
            # see: https://github.com/google/jax/issues/14003
            if jnp.isinf(self.lower).any():
                lower = jnp.nan_to_num(self.lower)
            if jnp.isinf(self.upper).any():
                upper = jnp.nan_to_num(self.upper)
            return jax.random.uniform(
                key, self.shape, minval=lower, maxval=upper, dtype=self.dtype
            )
        else:
            raise NotImplementedError(
                "Cannot sample from space of type {}".format(self.dtype)
            )


def is_discrete(space: Space) -> bool:
    return isinstance(space, Discrete)
