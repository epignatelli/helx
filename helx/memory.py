from __future__ import annotations

from collections import deque

import jax

from .mdp import Episode, tree_stack


class ReplayBuffer:
    """A replay buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    This type of buffer is usually used
    with off-policy methods, such as DQN or ACER.
    Note that, to save memory, it stores only the two extremes
    of the trajectory and accumulates the discunted rewards
    at each time step, to calculate the value target.
    However, this does not allow for off-policy corrections with nstep methods
    at consumption time. To perform off-policy corrections, please store
    the action probabilities foreach time step in the buffer.
    """

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.key = jax.random.PRNGKey(seed)
        self.elements = [None] * capacity
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.elements[idx]

    def full(self) -> bool:
        return len(self) == self.capacity

    def add(
        self,
        transition: object,
    ) -> None:
        idx = self.length % self.capacity
        self.elements[idx] = transition  # type: ignore
        self.length += 1

    def add_range(self, elements) -> None:
        for element in elements:
            self.add(element)

    def sample(self, n: int) -> Episode:
        self.key, k = jax.random.split(self.key)
        n = min(n, self.length)
        indices = jax.random.randint(k, (n,), 0, len(self))
        return tree_stack([self.elements[i] for i in indices])  # type: ignore
