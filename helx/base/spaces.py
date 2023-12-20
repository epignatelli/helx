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
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.random import KeyArray

from jax.core import Shape
from jax import Array
from jax.core import ShapedArray, Shape


MIN_INT = jax.numpy.iinfo(jnp.int16).min
MAX_INT = jax.numpy.iinfo(jnp.int16).max
MIN_INT_ARR = jnp.asarray(MIN_INT)
MAX_INT_ARR = jnp.asarray(MAX_INT)


class Space(ShapedArray):
    shape: Sequence[int]
    minimum: int = MIN_INT
    maximum: int = MAX_INT

    def __repr__(self):
        return "{}, min={}, max={})".format(
            super().__repr__()[:-1], self.minimum, self.maximum
        )

    def sample(self, key: KeyArray) -> Array:
        raise NotImplementedError()


class Discrete(Space):
    def __init__(self, n_elements: int = MAX_INT, shape: Shape = (), dtype=jnp.int32):
        super().__init__(shape, dtype)
        self.minimum = 0
        self.maximum = n_elements - 1

    def sample(self, key: KeyArray) -> Array:
        item = jax.random.randint(key, self.shape, self.minimum, self.maximum)
        # randint cannot draw jnp.uint, so we cast it later
        return jnp.asarray(item, dtype=self.dtype)


class Continuous(Space):
    def __init__(
        self, shape: Shape = (), minimum: int = MIN_INT, maximum: int = MAX_INT
    ):
        super().__init__(shape, jnp.float32)
        self.minimum = minimum
        self.maximum = maximum

    def sample(self, key: KeyArray) -> Array:
        assert jnp.issubdtype(self.dtype, jnp.floating)
        # see: https://github.com/google/jax/issues/14003
        lower = jnp.nan_to_num(self.minimum)
        upper = jnp.nan_to_num(self.maximum)
        return jax.random.uniform(
            key, self.shape, minval=lower, maxval=upper, dtype=self.dtype
        )
