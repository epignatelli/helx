import logging
from collections import deque
from typing import NamedTuple, Sequence

import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as onp

from ..typing import Key


class Transition(NamedTuple):
    x_0: onp.ndarray  #  observation at t=0
    a_0: onp.ndarray  #  actions at t=0
    r_0: onp.ndarray  #  rewards at t=1
    x_1: onp.ndarray  # observatin at t=n (note multistep)
    gamma: onp.ndarray = 1.0  #  discount factor
    trace_decay: onp.ndarray = 1.0  # trace decay for lamba returns


class OfflineBuffer:
    """A replay buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    This type of buffer is usually used
    with off-policy methods, such as DQN or ACER.
    """

    def __init__(
        self,
        capacity: int,
        n_steps: int = 1,
        cumulative: bool = True,
        seed: int = 0,
    ):
        #  public:
        self.capacity = capacity
        self.n_steps = n_steps
        self.cumulative = cumulative
        self.seed = seed

        #  private:
        self._rng = jax.random.PRNGKey(seed)
        self._episodes = deque(maxlen=capacity)
        self._current_transition = None
        self._t = 0

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, idx):
        return self._episodes[idx]

    def full(self):
        return len(self) == self.capacity

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
        trace_decay: float = 1.0,
    ) -> None:
        #  start of a new episode
        if timestep.first() or self._t == 0:
            self._current_transition = Transition(
                x_0=timestep.observation,
                a_0=int(action),
                r_0=float(new_timestep.reward),
                x_1=None,
            )
        elif timestep.mid():
            # accumulate rewards
            self._current_transition._replace(
                r_0=self._current_transition.r_0
                + new_timestep.reward * new_timestep.discount
            )

        self._t += 1
        if (new_timestep.last()) or (self._t >= self.n_steps):
            self._current_transition._replace(x_1=new_timestep.observation)
            self._episodes.append(self._current_transition)
            self._t = 0
        return

    def sample(self, n: int, rng: Key = None) -> Transition:
        high = len(self) - n
        if high <= 0:
            logging.warning(
                "The buffer contains less elements than requested: {} <= {}\n"
                "Returning all the available elements".format(len(self), n)
            )
            indices = range(len(self))
        elif rng is None:
            rng, _ = jax.random.split(self._rng)
        indices = jax.random.randint(rng, (n,), 0, high)

        x_0 = jnp.array([self._episodes[idx].x_0 for idx in indices]).astype(
            jnp.float32
        )
        a_0 = jnp.array([self._episodes[idx].a_0 for idx in indices])
        r_0 = jnp.array([self._episodes[idx].r_0 for idx in indices])
        x_1 = jnp.array([self._episodes[idx].x_1 for idx in indices]).astype(
            jnp.float32
        )
        return Transition(x_0, a_0, r_0, x_1)


class OnlineBuffer(Sequence):
    """A replay buffer that store a single n-step trajectory
    of experience.
    This type of buffer is usually used with online methods,
    generally on-policy methods, such as A2C.
    """

    def __init__(
        self,
        capacity: int,
        observation_spec: specs.Array,
        n_steps: int = 1,
    ):
        #  public:
        self.capacity = capacity
        self.n_steps = n_steps
        self.trajectory = Transition(
            x_0=jnp.empty(n_steps, *observation_spec.shape),
            a_0=jnp.empty(n_steps - 1, 1),
            r_0=jnp.empty(n_steps - 1, 1),
            x_1=jnp.empty(n_steps, *observation_spec.shape),
        )

        #  private:
        self._t = 0
        self._terminal = False

    def full(self):
        return (self._t == self.n_steps - 1) or (self._terminal)

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        # set buffer to not result as full
        self._terminal = new_timestep.last()
        # collect experience
        self.trajectory.x_0[self._t] = jnp.array(
            timestep.observation, dtype=jnp.float32
        )
        self.trajectory.a_0[self._t] = int(action)
        self.trajectory.r_0[self._t] = float(new_timestep.reward)
        self.trajectory.x_1[self._t] = jnp.array(
            new_timestep.observation, dtype=jnp.float32
        )
        # prepare to new experience
        self._t += 1
        # if episode is terminal or we reached T, we reset the trajectory
        if self.full():
            self._t = 0
        return

    def sample(self) -> Transition:
        return self.trajectory
