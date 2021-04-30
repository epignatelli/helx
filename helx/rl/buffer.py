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
    Note that, to save memory, it stores only the two extremes
    of the trajectory and accumulates the discunted rewards
    at each time step, to calculate the value target.
    However, this does not allow for off-policy corrections with nstep methods
    at consumption time. To perform off-policy corrections, please store
    the action probabilities foreach time step in the buffer.
    """

    def __init__(
        self,
        capacity: int,
        n_steps: int = 1,
        seed: int = 0,
    ):
        #  public:
        self.capacity = capacity
        self.n_steps = n_steps
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
                + (new_timestep.discount ** self._t) * new_timestep.reward
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


class Trajectory(NamedTuple):
    observations: onp.ndarray  #  observation at t=0 and t=1
    actions: onp.ndarray  #  actions at t=0
    rewards: onp.ndarray  #  rewards at t=1
    gammas: onp.ndarray = 1.0  #  discount factor
    lambdas: onp.ndarray = 1.0  #  discount factor


class OnlineBuffer(Sequence):
    """A replay buffer that stores a single n-step trajectory
    of experience.
    This type of buffer is most commonly used with online methods,
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
        self.observation_spec = observation_spec
        self.n_steps = n_steps

        #  private:
        self._reset()

    def full(self):
        return (self._t == self.n_steps - 1) or (self._terminal)

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        # collect experience
        self.trajectory.observations[self._t] = jnp.array(
            timestep.observation, dtype=jnp.float32
        )
        self.trajectory.actions[self._t] = int(action)
        self.trajectory.rewards[self._t] = float(new_timestep.reward)
        self.trajectory.gammas[self._t] = float(timestep.discount)

        # update buffer state
        self._t += 1
        self._terminal = new_timestep.last()

        # if the trajectory cannot move forwards, add the last observation
        if self.full():
            self._reset()
        return

    def sample(self) -> Transition:
        return self.trajectory

    def _reset(self):
        self._t = 0
        self.trajectory = Trajectory(
            observations=jnp.empty(self.n_steps + 1, *self.observation_spec.shape),
            actions=jnp.empty(self.n_steps, 1),
            rewards=jnp.empty(self.n_steps, 1),
            gammas=jnp.empty(self.n_steps, 1),
        )
        self._terminal = False
        return
