import logging
from collections import deque
from typing import Callable, NamedTuple, Sequence

import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp

from ..typing import Action, Discount, Key, Observation, Reward, Batch, TraceDecay


class Transition(NamedTuple):
    observation: Observation  #  observation at t=0
    action: Action  #  actions at t=0
    reward: Reward  #  rewards at t=1
    new_observation: Observation  # observatin at t=n (note multistep)
    discount: Discount = 1.0  #  discount factor
    trace_decay: TraceDecay = 1.0  # trace decay for lamba returns


class Trajectory(NamedTuple):
    observations: Batch[Observation]  #  observation at t=0 and t=1
    actions: Batch[Action]  #  actions at t=0
    rewards: Batch[Reward]  #  rewards at t=1
    discounts: Batch[Discount] = None  #  discount factor
    trace_decays: Batch[TraceDecay] = None  #  discount factor


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
        preprocess=lambda x: x,
    ) -> None:
        #  start of a new episode
        if timestep.first() or self._t == 0:
            self._current_transition = Transition(
                observation=preprocess(timestep.observation),
                action=int(action),
                reward=float(new_timestep.reward),
                new_observation=None,
            )
        elif timestep.mid():
            # accumulate rewards
            self._current_transition._replace(
                reward=self._current_transition.reward
                + (new_timestep.discount ** self._t) * new_timestep.reward
            )

        self._t += 1
        if (new_timestep.last()) or (self._t >= self.n_steps):
            self._current_transition._replace(
                new_observation=preprocess(new_timestep.observation)
            )
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

        observation = jnp.array(
            [self._episodes[idx].observation for idx in indices]
        ).astype(jnp.float32)
        action = jnp.array([self._episodes[idx].action for idx in indices])
        reward = jnp.array([self._episodes[idx].reward for idx in indices])
        new_observation = jnp.array(
            [self._episodes[idx].new_observation for idx in indices]
        ).astype(jnp.float32)
        return Transition(observation, action, reward, new_observation)


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
        return self._t == self.n_steps - 1

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep = None,
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
        self.trajectory.gammas[self._t] = float(new_timestep.discount)
        self._t += 1

        #  if we have enough transitions, add last obs and return
        if self.full():
            self.trajectory.observations[self._t] = preprocess(
                jnp.array(new_timestep.observation, dtype=jnp.float32)
            )
            return
        #  if we do not have enough transitions, and can't sample more, retry
        if new_timestep.last():
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
            discounts=jnp.empty(self.n_steps, 1),
            trace_decays=jnp.empty(self.n_steps, 1),
        )
        return
