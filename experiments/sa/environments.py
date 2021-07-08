from functools import wraps
from bsuite.environments import catch as bs_catch
import dm_env


Catch = bs_catch.Catch


class DelayedCatch(bs_catch.Catch):
    """A version of the Catch environment when rewards are
    delivered only at the end of an episode"""

    @wraps(bs_catch.Catch.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = super()
        self.reward = 0.0

    @wraps(bs_catch.Catch.step)
    def step(self, action: int) -> dm_env.TimeStep:
        timestep = self.base.step(action)
        #  cache rewards
        self.reward += timestep.reward
        #  return rewards if the episode terminates, otherwise return 0
        reward = self.reward * int(timestep.last())
        return timestep._replace(reward=reward)

    @wraps(bs_catch.Catch.reset)
    def reset(self) -> dm_env.TimeStep:
        self.rewards = 0.0
        timestep = self.base.reset()
        return timestep._replace(reward=0.0)
