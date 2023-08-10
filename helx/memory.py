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

from typing import Any, List, Tuple
from flax import struct

import jax
from jax.random import KeyArray
import jax.numpy as jnp

from .spaces import Space
from .mdp import Timestep, TRANSITION


class ReplayBuffer(struct.PyTreeNode):
    """A circular buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    Use the `CircularBuffer.init` method to construct a buffer."""

    capacity: int = struct.field(pytree_node=False)
    """Returns the capacity of the buffer."""
    elements: Any = struct.field(pytree_node=True)
    """The elements currently stored in the buffer."""
    idx: jax.Array = struct.field(pytree_node=True)
    """The index of the next element to be added to the buffer."""

    @classmethod
    def create(cls, timestep: Timestep, capacity: int, n_steps: int=1) -> ReplayBuffer:
        """Constructs a CircularBuffer class."""
        # reserve memory
        uninitialised_elements = jax.tree_map(
            lambda x: jnp.broadcast_to(
                jnp.asarray(x * 0, dtype=x.dtype),
                (capacity, n_steps + 1, *jnp.asarray(x).shape),
            ),
            timestep,
        )
        return cls(
            capacity=capacity,
            elements=uninitialised_elements,
            idx=jnp.asarray(0),
        )

    def size(self) -> jax.Array:
        """Returns the number of elements currently stored in the buffer."""
        return self.idx

    def add(self, item: Any) -> ReplayBuffer:
        """Adds a single element to the buffer. If the buffer is full,
        the oldest element is overwritten."""
        idx = self.idx % self.capacity
        elements = jax.tree_map(lambda x, y: x.at[idx].set(y), self.elements, item)
        return self.replace(
            idx=self.idx + 1,
            elements=elements,
        )

    def sample(self, key: KeyArray, n: int = 1) -> Any:
        """Samples `n` elements uniformly at random from the buffer,
        and stacks them into a single pytree.
        If `n` is greater than state.idx,
        the function returns uninitialised elements"""
        indices = jax.random.randint(key=key, shape=(n,), minval=0, maxval=self.idx)
        items = jax.tree_map(lambda x: x[indices], self.elements)
        return items


class EpisodeBuffer(struct.PyTreeNode):
    """A asynchronous episodic memory buffer used for online learning."""

    elements: Any
    """The elements currently stored in the buffer."""
    idx: int = struct.field(pytree_node=False)
    """The index of the next element to be added to the buffer."""
    size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, example_item: Any, size: int) -> EpisodeBuffer:
        """Constructs a CircularBuffer class."""
        # reserve memory
        uninitialised_elements = jax.tree_map(
            lambda x: jnp.broadcast_to(x * 0, (size, *x.shape)), example_item
        )
        return cls(elements=uninitialised_elements, idx=0, size=size)

    def add(self, item: Any) -> EpisodeBuffer:
        """Adds a single element to the buffer. If the buffer is full,
        the oldest element is overwritten."""
        # index updating requires jitting to guarantee in-place efficiency
        # see https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
        elements = jax.tree_map(lambda x, y: x.at[self.idx].set(y), self.elements, item)
        return self.replace(elements=elements, idx=self.idx + 1)

    def add_range(self, items: List[Any]) -> EpisodeBuffer:
        """Adds more than one elements to the buffer. If the buffer is full,
        the oldest elements are overwritten."""
        start = self.idx % self.size
        end = start + len(items)
        # index updating requires jitting to guarantee in-place efficiency
        # see https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
        elements = jax.tree_map(
            lambda x, y: x.at[start:end].set(jnp.stack(y)), self.elements, items
        )
        return self.replace(elements=elements, idx=self.idx + len(items))

    def sample(self, n: int = 1) -> Tuple[Any, EpisodeBuffer]:
        """Samples `n` elements uniformly at random from the buffer,
        and stacks them into a single pytree. If `n` is greater than state.idx,
        the function returns uninitialised elements"""
        indices = jnp.arange(self.idx)[:n]
        items = jax.tree_map(lambda x: x[indices], self.elements)
        return items, self
