from __future__ import annotations

from typing import cast

import gym
import gym.core
import gym.utils.seeding
import jax
import jax.numpy as jnp
from chex import Array
import numpy as np

from .base import Environment
from ..mdp import Action, StepType, Timestep
from ..spaces import Continuous, Space


class FromGymEnv(Environment[gym.Env]):
    """Static class to convert between gym and helx environments."""

    def __init__(self, env: gym.core.Env):
        super().__init__(env)

    def action_space(self) -> Space:
        if self._action_space is not None:
            return self._action_space

        self._action_space = Space.from_gym(self._env.action_space)
        return self._action_space

    def observation_space(self) -> Space:
        if self._observation_space is not None:
            return self._observation_space

        self._observation_space = Space.from_gym(self._env.observation_space)
        return self._observation_space

    def reward_space(self) -> Space:
        if self._reward_space is not None:
            return self._reward_space

        minimum = self._env.reward_range[0]
        maximum = self._env.reward_range[1]
        self._reward_space = Continuous((1,), (minimum,), (maximum,))
        return self._reward_space

    def state(self) -> Array:
        if self._current_observation is None:
            raise ValueError(
                "Environment not initialized. Run `reset` first, to set a starting state."
            )
        return self._current_observation

    def reset(self, seed: int | None = None) -> Timestep:
        try:
            obs, _ = self._env.reset(seed=seed)
        except TypeError:
            # TODO(epignatelli): remove try/except when gym3 is updated.
            # see: https://github.com/openai/gym3/issues/8
            obs, _ = self._env.reset()
        self._current_observation = jnp.asarray(obs)
        return Timestep(obs, None, StepType.TRANSITION)

    def step(self, action: Action) -> Timestep:
        action_ = np.asarray(action)
        next_step = self._env.step(action_)
        self._current_observation = jnp.asarray(next_step[0])
        gym.core.ObsType
        return Timestep.from_gym(next_step)

    def seed(self, seed: int) -> None:
        self._env.np_random, seed = gym.utils.seeding.np_random(seed)
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human"):
        self._env.render_mode = mode
        return self._env.render()

    def close(self) -> None:
        return self._env.close()

    def name(self) -> str:
        return self._env.unwrapped.__class__.__name__
