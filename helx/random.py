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


"""Random number generation utilities."""
from typing import Generator
import jax
import jax.numpy as jnp

from jax.random import KeyArray as Key


KeySequence = Generator[Key, Key, None]


def shuffled_batched_indices(
    rng: Key,
    stream_len: int,
    batch_size: int,
    drop_last: bool = False,
):
    if isinstance(stream_len, list):
        #  stream_len is a sequence of indices already, or a list of objects
        stream_len = len(stream_len)
    shuffled = jax.random.permutation(rng, jnp.arange(0, stream_len))
    shuffled_batched = jnp.array_split(
        shuffled,
        jnp.arange(batch_size, stream_len, batch_size),
    )
    if stream_len % batch_size and drop_last:
        shuffled_batched = shuffled_batched[:-1]
    return shuffled_batched


def PRNGSequence(seed: int) -> KeySequence:
    k = jax.random.PRNGKey(seed)
    while True:
        rng, k = jax.random.split(k)
        yield rng


def new_key(seq: KeySequence) -> Key:
    return next(seq)
