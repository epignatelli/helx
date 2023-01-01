"""A set of functions to interoperate between the most common
RL environment interfaces, like `gym`, `gymnasium`, `dm_env`, `bsuite and others."""
from __future__ import annotations

import abc
from typing import Any, cast

import bsuite.environments
import dm_env
import gym.core
import gymnasium
import gymnasium.core
import gymnasium.utils.seeding
import jax
import jax.numpy as jnp
from chex import Array, Shape

from .mdp import Action, StepType, Timestep, GymnasiumTimestep, GymTimestep
from .spaces import BoundedRange, Space


class Environment(abc.ABC):
    def __init__(self):
        self._action_space = None
        self._observation_space = None
        self._reward_space = None
        self._current_observation = None
        self._seed = 0
        self._key = jax.random.PRNGKey(self._seed)

    @abc.abstractmethod
    def action_space(self) -> Space:
        ...

    @abc.abstractmethod
    def observation_space(self) -> Space:
        ...

    @abc.abstractmethod
    def reward_space(self) -> Space:
        ...

    @abc.abstractmethod
    def state(self) -> Array:
        ...

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> Timestep:
        ...

    @abc.abstractmethod
    def step(self, action: int) -> Timestep:
        ...

    @abc.abstractmethod
    def seed(self, seed: int) -> None:
        ...

    @abc.abstractmethod
    def render(self, mode: str = "human"):
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @staticmethod
    def make(env: Any):
        if isinstance(env, gymnasium.core.Env):
            return FromGymnasiumEnv(env)
        elif isinstance(env, gym.core.Env):
            return FromGymEnv(env)
        elif isinstance(env, dm_env.Environment):
            return FromDmEnv(env)
        elif isinstance(env, bsuite.environments.Environment):
            return FromBsuiteEnv(env)
        else:
            raise TypeError(
                f"Environment type {type(env)} is not supported. "
                "Only gymnasium, gym, dm_env and bsuite environments are supported."
            )

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __del__(self):
        self.close()


class FromGymnasiumEnv(Environment):
    """Static class to convert between gymnasium and helx environments."""

    def __init__(self, env: gymnasium.Env):
        super().__init__()
        self._env: gymnasium.Env = env

    def action_space(self) -> Space:
        if self._action_space is not None:
            return self._action_space

        self._action_space = Space.from_gymnasium(self._env.action_space)
        return self._action_space

    def observation_space(self) -> Space:
        if self._observation_space is not None:
            return self._observation_space

        self._observation_space = Space.from_gymnasium(self._env.observation_space)
        return self._observation_space

    def reward_space(self) -> Space:
        if self._reward_space is not None:
            return self._reward_space

        minimum = self._env.reward_range[0]
        maximum = self._env.reward_range[1]
        self._reward_space = BoundedRange(minimum, maximum)
        return self._reward_space

    def state(self) -> Array:
        if self._current_observation is None:
            raise ValueError(
                "Environment not initialized. Run `reset` first, to set a starting state."
            )
        return self._current_observation

    def reset(self, seed: int | None = None) -> Timestep:
        obs, info = self._env.reset(seed=seed)
        self._current_observation = jnp.asarray(obs)
        return Timestep(obs, None, StepType.TRANSITION)

    def step(self, action: Action) -> Timestep:
        next_step = cast(GymnasiumTimestep, self._env.step(action))
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_gymnasium(next_step)

    def seed(self, seed: int) -> None:
        self._env.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human"):
        self._env.render_mode = mode
        return self._env.render()

    def close(self) -> None:
        return self._env.close()


class FromGymEnv(Environment):
    """Static class to convert between gymnasium and helx environments."""

    def __init__(self, env: gym.core.Env):
        super().__init__()
        self._env: gym.core.Env = env

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
        self._reward_space = BoundedRange(minimum, maximum)
        return self._reward_space

    def state(self) -> Array:
        if self._current_observation is None:
            raise ValueError(
                "Environment not initialized. Run `reset` first, to set a starting state."
            )
        return self._current_observation

    def reset(self, seed: int | None = None) -> Timestep:
        obs, info = self._env.reset(seed=seed)
        self._current_observation = jnp.asarray(obs)
        return Timestep(obs, None, StepType.TRANSITION)

    def step(self, action: Action) -> Timestep:
        next_step = self._env.step(action)
        self._current_observation = jnp.asarray(next_step[0])
        return Timestep.from_gym(next_step)

    def seed(self, seed: int) -> None:
        self._env.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human"):
        self._env.render_mode = mode
        return self._env.render()

    def close(self) -> None:
        return self._env.close()


class FromDmEnv(Environment):
    """Static class to convert between dm_env and helx environments."""

    def __init__(self, env: dm_env.Environment):
        super().__init__()
        self._env = env

    def action_space(self) -> Space:
        if self._action_space is not None:
            return self._action_space

        # TODO (epignatelli): remove type ignore once dm_env is correctly typed.
        self._action_space = Space.from_dm_env(self._env.action_spec)  # type: ignore
        return self._action_space

    def observation_space(self) -> Space:
        if self._observation_space is not None:
            return self._observation_space

        # TODO (epignatelli): remove type ignore once dm_env is correctly typed.
        self._observation_space = Space.from_dm_env(self._env.observation_spec)  # type: ignore
        return self._observation_space

    def reward_space(self) -> Space:
        if self._reward_space is not None:
            return self._reward_space

        # TODO (epignatelli): remove type ignore once dm_env is correctly typed.
        self._reward_space = Space.from_dm_env(self._env.reward_spec)  # type: ignore
        return self._reward_space

    def state(self) -> Array:
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
        next_step = self._env.step(action)
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


class FromBsuiteEnv(Environment):
    """Static class to convert between bsuite.Environment and helx environments."""

    def __init__(self, env: bsuite.environments.Environment):
        super().__init__()
        self._env = env

    def action_space(self) -> Space:
        if self._action_space is not None:
            return self._action_space

        # TODO (epignatelli): Remove this once dm_env is correctly typed.
        self._action_space = Space.from_dm_env(self._env.action_spec)  # type: ignore
        return self._action_space

    def observation_space(self) -> Space:
        if self._observation_space is not None:
            return self._observation_space

        # TODO (epignatelli): Remove this once dm_env is correctly typed.
        self._observation_space = Space.from_dm_env(self._env.observation_spec)  # type: ignore
        return self._observation_space

    def reward_space(self) -> Space:
        if self._reward_space is not None:
            return self._reward_space

        # TODO (epignatelli): Remove this once dm_env is correctly typed.
        self._reward_space = Space.from_dm_env(self._env.reward_spec)  # type: ignore
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
        # TODO (epignatelli): Remove this once bsuite is correctly typed
        next_step = self._env.step(action)  # type: ignore
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


class FromGymnaxEnv(Environment):
    """Static class to convert between Gymnax environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()


class FromIvyGymEnv(Environment):
    """Static class to convert between Ivy Gym environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()


class FromDMControlEnv(Environment):
    """Static class to convert between dm_control environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()


class FromMujocoEnv(Environment):
    """Static class to convert between mujoco environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()
