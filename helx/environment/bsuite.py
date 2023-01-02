from __future__ import annotations

import bsuite.environments
import jax
import jax.numpy as jnp
from chex import Array

from ..mdp import Action, Timestep
from ..spaces import Space
from .base import Environment


class FromBsuiteEnv(Environment[bsuite.environments.Environment]):
    """Static class to convert between bsuite.Environment and helx environments."""

    def __init__(self, env: bsuite.environments.Environment):
        super().__init__(env)

    def action_space(self) -> Space:
        if self._action_space is not None:
            return self._action_space

        # TODO (epignatelli): Remove this once dm_env is correctly typed.
        self._action_space = Space.from_dm_env(self._env.action_spec())  # type: ignore
        return self._action_space

    def observation_space(self) -> Space:
        if self._observation_space is not None:
            return self._observation_space

        # TODO (epignatelli): Remove this once dm_env is correctly typed.
        self._observation_space = Space.from_dm_env(self._env.observation_spec())  # type: ignore
        return self._observation_space

    def reward_space(self) -> Space:
        if self._reward_space is not None:
            return self._reward_space

        # TODO (epignatelli): Remove this once dm_env is correctly typed.
        self._reward_space = Space.from_dm_env(self._env.reward_spec())  # type: ignore
        return self._reward_space

    def state(self) -> Array:
        # TODO (epignatelli): Remove this once bsuite is updated.
        if hasattr(self._env, "_get_observation"):
            return self._env._get_observation()  # type: ignore

        if self._current_observation is None:
            raise ValueError(
                "Environment not initialized. Run `reset` first to produce a starting state."
            )
        return self._current_observation

    def reset(self, seed: int | None = None) -> Timestep:
        next_step = self._env.reset()
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_dm_env(next_step)

    def step(self, action: Action) -> Timestep:
        # TODO (epignatelli): Remove `type: ignore` once bsuite is correctly typed
        next_step = self._env.step(action.item())
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_dm_env(next_step)

    def seed(self, seed: int) -> None:
        self._seed = seed
        self._key = jax.random.PRNGKey(self._seed)
        return

    def render(self, mode: str = "human"):
        # TODO: Handle mode
        current_state = self.state()
        return current_state

    def close(self) -> None:
        return self._env.close()
