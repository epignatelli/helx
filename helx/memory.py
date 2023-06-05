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

from typing import Any, List, Tuple
from flax import struct

import jax
from jax.random import KeyArray
import jax.numpy as jnp


class ReplayBuffer(struct.PyTreeNode):
    """A circular buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    Use the `CircularBuffer.init` method to construct a buffer."""

    elements: Any
    """The elements currently stored in the buffer."""
    capacity: int = struct.field(pytree_node=False)
    """Returns the capacity of the buffer."""
    idx: int = struct.field(pytree_node=False)
    """The index of the next element to be added to the buffer."""

    @classmethod
    def create(cls, example_item: Any, capacity: int) -> ReplayBuffer:
        """Constructs a CircularBuffer class."""
        # reserve memory
        uninitialised_elements = jax.tree_map(
            lambda x: jnp.broadcast_to(x * 0, (capacity, *jnp.asarray(x).shape)), example_item
        )
        return cls(
            elements=uninitialised_elements,
            capacity=capacity,
            idx=0,

        )

    def size(self):
        """Returns the number of elements currently stored in the buffer."""
        return self.idx

    def is_full(self):
        """Returns whether the buffer has reached its full capacity."""
        return self.idx == self.capacity

    def add(self, item: Any) -> ReplayBuffer:
        """Adds a single element to the buffer. If the buffer is full,
        the oldest element is overwritten."""
        idx = self.idx % self.capacity
        # index updating requires jitting to guarantee in-place efficiency
        # see https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
        elements = jax.tree_map(lambda x, y: x.at[idx].set(y), self.elements, item)
        return self.replace(
            idx=self.idx + 1,
            elements=elements,
        )

    def add_range(self, items: List[Any]) -> ReplayBuffer:
        """Adds more than one elements to the buffer. If the buffer is full,
        the oldest elements are overwritten."""
        start = self.idx % self.capacity
        end = start + len(items)
        # index updating requires jitting to guarantee in-place efficiency
        # see https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
        stacked_items = jax.tree_map(lambda *x: jnp.stack(x).squeeze(), *items)
        elements = jax.tree_map(
            lambda x, y: x.at[start:end].set(y), self.elements, stacked_items
        )
        return self.replace(
            idx=self.idx + len(items),
            elements=elements,
        )

    def sample(self,  key: KeyArray, n: int = 1) -> Tuple[ReplayBuffer, Any]:
        """Samples `n` elements uniformly at random from the buffer,
        and stacks them into a single pytree.
        If `n` is greater than state.idx,
        the function returns uninitialised elements"""
        indices = jax.random.randint(key=key, shape=(n,), minval=0, maxval=self.idx)
        items = jax.tree_map(lambda x: x[indices], self.elements)
        buffer = self.replace(key=key)
        return buffer, items


class EpisodeBuffer(struct.PyTreeNode):
    """A asynchronous episodic memory buffer used for online learning."""
    elements: Any
    """The elements currently stored in the buffer."""
    idx: int = struct.field(pytree_node=False)
    """The index of the next element to be added to the buffer."""
    size: int = struct.field(pytree_node=False)

    @classmethod
    def init(cls, example_item: Any, size: int) -> EpisodeBuffer:
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



# class Buffer(Generic[PyTreeDef]):
#     """A circular buffer used for Experience Replay (ER):
#     Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
#     This type of buffer is usually used
#     with off-policy methods, such as DQN or ACER.
#     Note that, to save memory, it stores only the two extremes
#     of the trajectory and accumulates the discunted rewards
#     at each time step, to calculate the value target.
#     However, this does not allow for off-policy corrections with nstep methods
#     at consumption time. To perform off-policy corrections, please store
#     the action probabilities foreach time step in the buffer.

#     Args:
#         capacity (int): The maximum number of elements that can be stored in the buffer.
#         seed (int): The seed used to initialise the random number generator for sampling.
#     """

#     def __init__(self, capacity: int, seed: int = 0):
#         self.elements: Deque[PyTreeDef] = deque(maxlen=capacity)
#         self.key: KeyArray = jax.random.PRNGKey(seed)

#     def __len__(self):
#         return len(self.elements)

#     def __getitem__(self, idx):
#         return self.elements[idx]

#     @property
#     def capacity(self) -> int:
#         """Returns the capacity of the buffer."""
#         return self.elements.maxlen or 0

#     def full(self) -> bool:
#         """Returns whether the buffer has reached its capacity."""
#         return len(self) == self.capacity

#     def add(self, transition: PyTreeDef) -> None:
#         """Adds an element to the buffer. If the buffer is full, the oldest element
#         is overwritten."""
#         self.elements.append(transition)

#     def add_range(self, elements: Sequence[PyTreeDef]) -> None:
#         """Adds more than one elements to the buffer.
#         Args:
#             elements (Sequence[PyTreeDef]): A sequence of elements to add to the buffer.
#         """
#         for element in elements:
#             self.elements.append(element)

#     def sample(self, n: int) -> PyTreeDef:
#         """Samples `n` elements from the buffer, and stacks them into a single pytree
#         to form a batch of `PyTreeDef` elements.
#         Args:
#             n (int): The number of elements to sample.
#         Returns:
#             PyTreeDef: A pytree of `n` elements, where each element in the pytree has an
#             additional batch dimension.
#         """
#         self.key, k = jax.random.split(self.key)
#         n = min(n, len(self.elements))
#         indices = jax.random.randint(k, (n,), 0, len(self))
#         return tree_stack([self.elements[i] for i in indices])
