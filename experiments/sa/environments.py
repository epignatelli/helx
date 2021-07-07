from functools import wraps
from bsuite.environments import catch as bs_catch
import dm_env


Catch = bs_catch.Catch


class DelayedCatch(bs_catch.Catch):
    @wraps(bs_catch.Catch.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = super()
        self.reward = 0.0

    @wraps(bs_catch.Catch.step)
    def step(self, action: int) -> dm_env.TimeStep:
        timestep = self.bases.step(action)
        self.reward += timestep.reward
        return timestep._replace(reward=0.0)

    @wraps(bs_catch.Catch.reset)
    def reset(self) -> dm_env.TimeStep:
        self.rewards = 0.0
        timestep = self.base.reset()
        return timestep._replace(reward=0.0)
