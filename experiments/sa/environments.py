from functools import wraps
from bsuite.environments import catch as bs_catch
import dm_env
import numpy as np


class Catch(bs_catch.Catch):
    def __init__(self, seed: int):
        super().__init__(7, 7, seed)


class DelayedCatch(Catch):
    """A version of the Catch environment when rewards are
    delivered only at the end of an episode"""

    @wraps(Catch.__init__)
    def __init__(self, seed: int):
        super().__init__(seed)
        self._base = super()
        self._reward = 0.0
        self._current_inner_episode: int = 0
        self._max_inner_episodes: int = 20

    def _is_final(self):
        return self._current_inner_episode >= self._max_inner_episodes

    @wraps(Catch.reset)
    def _reset(self) -> dm_env.TimeStep:
        self._rewards = 0.0
        self._current_inner_episode = 0
        return self._base.reset()

    @wraps(Catch.step)
    def _step(self, action: int) -> dm_env.TimeStep:
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        # Move the paddle.
        dx = bs_catch._ACTIONS[action]
        self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

        # Drop the ball.
        self._ball_y += 1

        # Check for termination.
        reward = 0.0
        if self._ball_y == self._paddle_y:
            reward = 1.0 if self._paddle_x == self._ball_x else -1.0
            self._reward += reward
            self._reset_next_step = True
            self._total_regret += 1.0 - reward

        self._reward += reward
        if self.is_final():
            return dm_env.termination(
                reward=self._reward, observation=self._observation()
            )
        else:
            return dm_env.transition(reward=0.0, observation=self._observation())
