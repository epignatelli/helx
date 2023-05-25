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

from typing import Any, Generic, List, Sequence, Tuple, TypeVar, Deque
from collections import deque
from chex import Array, PyTreeDef, dataclass

import jax
from jax.random import KeyArray
import jax.numpy as jnp
from dataclass_array import DataclassArray
from dataclass_array.typing import Array, FloatArray, IntArray, BoolArray

from .spaces import Space
from .mdp import tree_stack


T = TypeVar("T")


class Buffer(Generic[T]):
    """A circular buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    This type of buffer is usually used
    with off-policy methods, such as DQN or ACER.
    Note that, to save memory, it stores only the two extremes
    of the trajectory and accumulates the discunted rewards
    at each time step, to calculate the value target.
    However, this does not allow for off-policy corrections with nstep methods
    at consumption time. To perform off-policy corrections, please store
    the action probabilities foreach time step in the buffer.

    Args:
        capacity (int): The maximum number of elements that can be stored in the buffer.
        seed (int): The seed used to initialise the random number generator for sampling.
    """

    def __init__(self, capacity: int, seed: int = 0):
        self.elements: Deque[T] = deque(maxlen=capacity)
        self.key: KeyArray = jax.random.PRNGKey(seed)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        return self.elements[idx]

    @property
    def capacity(self) -> int:
        """Returns the capacity of the buffer."""
        return self.elements.maxlen or 0

    def full(self) -> bool:
        """Returns whether the buffer has reached its capacity."""
        return len(self) == self.capacity

    def add(self, transition: T) -> None:
        """Adds an element to the buffer. If the buffer is full, the oldest element
        is overwritten."""
        self.elements.append(transition)

    def add_range(self, elements: Sequence[T]) -> None:
        """Adds more than one elements to the buffer.
        Args:
            elements (Sequence[T]): A sequence of elements to add to the buffer.
        """
        for element in elements:
            self.elements.append(element)

    def sample(self, n: int) -> T:
        """Samples `n` elements from the buffer, and stacks them into a single pytree
        to form a batch of `T` elements.
        Args:
            n (int): The number of elements to sample.
        Returns:
            T: A pytree of `n` elements, where each element in the pytree has an
            additional batch dimension.
        """
        self.key, k = jax.random.split(self.key)
        n = min(n, len(self.elements))
        indices = jax.random.randint(k, (n,), 0, len(self))
        return tree_stack([self.elements[i] for i in indices])


# DataclassArray allows treating batches of dataclasses as a single dataclass
# see https://github.com/google-research/dataclass_array#usage
T = TypeVar("T", bound=DataclassArray)


@dataclass
class CircularBufferState(Generic[T]):
    """The state of a CircularBuffer containing the current index, the key used to sample
    elements from the buffer, and the elements currently stored in the buffer.
    Elements must be a DataclassArray, which allows treating batches of dataclasses
    as a single dataclass.
    See https://github.com/google-research/dataclass_array#usage"""

    idx: int
    """The index of the next element to be added to the buffer."""
    key: KeyArray
    """The key used to sample elements from the buffer."""
    elements: T
    """The elements currently stored in the buffer."""


@dataclass
class CircularBuffer:
    """A circular buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    Use the `CircularBuffer.init` method to construct a buffer."""

    capacity: int
    """Returns the capacity of the buffer."""

    @classmethod
    def init(
        cls, example_item: T, capacity: int, seed: int = 0
    ) -> Tuple[CircularBuffer, CircularBufferState]:
        """Constructs a CircularBuffer class."""
        # reserve memory
        uninitialised_elements = jax.tree_map(
            lambda x: jnp.broadcast_to(x * 0, (capacity, *x.shape)), example_item
        )
        # config
        buffer = cls(capacity=capacity)
        state = CircularBufferState(
            idx=0,
            key=jax.random.PRNGKey(seed),
            elements=uninitialised_elements,
        )
        return buffer, state

    def size(self, state: CircularBufferState):
        """Returns the number of elements currently stored in the buffer."""
        return state.idx % self.capacity

    def is_full(self, state: CircularBufferState):
        """Returns whether the buffer has reached its full capacity."""
        return state.idx == self.capacity

    def add(self, state: CircularBufferState, item: T) -> CircularBuffer:
        """Adds a single element to the buffer. If the buffer is full, the oldest element
        is overwritten."""
        idx = state.idx % self.capacity
        # index updating requires jitting to guarantee in-place efficiency
        # see https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
        elements = state.elements.at[idx].set(item)
        return state.replace(
            idx=state.idx + 1,
            elements=elements,
        )

    def add_range(self, state: CircularBufferState, items: T) -> CircularBuffer:
        """Adds more than one elements to the buffer. If the buffer is full, the oldest elements
        are overwritten."""
        start = state.idx % self.capacity
        end = start + len(items)
        # index updating requires jitting to guarantee in-place efficiency
        # see https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
        elements = state.elements.at[start:end].set(items)
        return state.replace(idx=state.idx + len(items), elements=elements)

    def sample(self, state: CircularBufferState, n: int) -> Tuple[T, CircularBuffer]:
        """Samples `n` elements uniformly at random from the buffer,
        and stacks them into a single pytree"""
        state.key, k = jax.random.split(state.key)
        size = self.size(state)
        n = min(n, size)
        indices = jax.random.randint(key=k, shape=(n,), minval=0, maxval=size)
        return state.elements[indices], state.replace(key=state.key)
