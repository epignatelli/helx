import abc
import logging
from collections import deque
from typing import Callable, NamedTuple

import dm_env
import jax
import jax.numpy as jnp
from dm_env import specs
from jaxlib.xla_extension import Device

from ..jax import device_array
from ..typing import Action, Batch, Discount, Key, Observation, Reward, TraceDecay
from ..random import PRNGSequence


class Transition(NamedTuple):
    """A (s, a, r, s', a', γ, λ) transition with discount and lambda factors"""

    s: Observation  #  observation at t=0
    a: Action  #  actions at t=0
    r: Reward  #  rewards at t=1
    s1: Observation  # observatin at t=1 (note multistep)
    a1: Action  # action at t=1
    g: Discount = 1.0  #  discount factor
    l: TraceDecay = 1.0  # trace decay for lamba returns


class Trajectory(NamedTuple):
    """A set of batched transitions"""

    observations: Batch[Observation]  #  [T + 1, *obs.shape]
    actions: Batch[Action]  #  [T, 1] if off-policy, [T + 1, 1] otherwise
    rewards: Batch[Reward]  #  [T, 1]
    discounts: Batch[Discount] = None  #  [T, 1]
    trace_decays: Batch[TraceDecay] = None  #  [T, 1]


class IBuffer(abc.ABC):
    @abc.abstractmethod
    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
        preprocess=lambda x: x,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def full(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
        preprocess=lambda x: x,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, n: int, rng: Key = None) -> Trajectory:
        raise NotImplementedError


class OfflineBuffer(IBuffer):
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
        device: Device = None,
    ):
        #  public:
        self.capacity = capacity
        self.n_steps = n_steps
        self.seed = seed
        self.device = device
        self.trajectories = deque(maxlen=capacity)

        #  private:
        self._rng = jax.random.PRNGKey(seed)
        self._reset()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def full(self) -> bool:
        return len(self) == self.capacity

    def collecting(self) -> bool:
        return self._t < self.n_steps

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
        preprocess=lambda x: x,
    ) -> None:
        #  start of a new episode
        if not self.collecting():
            self.trajectories.append(self._current)
            self._reset()

        # add new transition to the trajectory
        store = lambda x: device_array(x, device=self.device)
        self._current.observations[self._t] = preprocess(store(timestep.observation))
        self._current.actions[self._t] = store(int(action))
        self._current.rewards[self._t] = store(float(new_timestep.reward))
        self._current.discounts[self._t] = store(float(new_timestep.discount))
        self._t += 1

        # ready to store, just add final observation
        if not self.collecting():
            self._current.observations[self._t] = preprocess(
                jnp.array(timestep.observation, dtype=jnp.float32)
            )
        # if not enough samples, and we can't sample the env anymore, reset
        elif new_timestep.last():
            self._reset()
        return

    def sample(self, n: int, rng: Key = None, device: Device = None) -> Trajectory:
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

        collate = lambda x: device_array(x, device=device)
        obs = collate([self.trajectories[idx].observations for idx in indices])
        actions = collate([self.trajectories[idx].actions for idx in indices])
        rewards = collate([self.trajectories[idx].rewards for idx in indices])
        discounts = collate([self.trajectories[idx].discounts for idx in indices])
        # traces = collate([self.trajectories[idx].trace_decays for idx in indices])
        return Trajectory(obs, actions, rewards, discounts)

    def _reset(self):
        self._t = 0
        self._current = Trajectory(
            observations=[None] * (self.n_steps + 1),
            actions=[None] * self.n_steps,
            rewards=[None] * self.n_steps,
            discounts=[None] * self.n_steps,
            trace_decays=[None] * self.n_steps,
        )


class OnlineBuffer(IBuffer):
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

    def full(self) -> bool:
        return self._t == self.n_steps - 1

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
        trace_decay: TraceDecay = 1.0,
        preprocess: Callable[[Observation], Observation] = lambda x: x,
    ) -> None:
        #  if buffer is full, prepare for new trajectory
        if self.full():
            self._reset()

        # add new transition to the trajectory
        self.trajectory.observations[self._t] = preprocess(
            jnp.array(timestep.observation, dtype=jnp.float32)
        )
        self.trajectory.actions[self._t] = int(action)
        self.trajectory.rewards[self._t] = float(new_timestep.reward)
        self.trajectory.discounts[self._t] = float(new_timestep.discount)
        self.trajectory.trace_decays[self._t] = float(trace_decay)
        self._t += 1

        #  if we have enough transitions, add last obs and return
        if self.full():
            self.trajectory.observations[self._t] = preprocess(
                jnp.array(new_timestep.observation, dtype=jnp.float32)
            )
        #  if we do not have enough transitions, and can't sample more, retry
        elif new_timestep.last():
            self._reset()
        return

    def sample(self, n: int = 1, rng: Key = None) -> Trajectory:
        return self.trajectory

    def _reset(self):
        self._t = 0
        self.trajectory = Trajectory(
            observations=jnp.empty(self.n_steps + 1, *self.observation_spec.shape),
            actions=jnp.empty(self.n_steps, 1),
            rewards=jnp.empty(self.n_steps, 1),
            discounts=jnp.empty(self.n_steps, 1),
            trace_decays=jnp.empty(self.n_steps, 1),
        )
        return


class EpisodicMemory(IBuffer):
    def __init__(self, seed: int = 0):
        #  public:
        self.states = []

        #  private:
        self._terminal = False
        self._rng = PRNGSequence(seed)
        self._reset()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    def full(self) -> bool:
        return self._terminal

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
        preprocess: Callable = lambda x: x,
    ) -> None:
        #  if buffer is full, prepare for new trajectory
        if self.full():
            self._reset()

        #  collect experience
        self.states.append(preprocess(timestep.observation))
        self._terminal = new_timestep.last()

        #   if transition is terminal, append last state
        if self.full():
            self.states.append(preprocess(new_timestep.observation))
        return

    def sample(self, n: int, rng: Key = None) -> Trajectory:
        key = next(self._rng)
        indices = jax.random.randint(key, (n,), 0, len(self))
        return [self.states[idx] for idx in indices]

    def _reset(self):
        self._terminal = False
        self.states = []
