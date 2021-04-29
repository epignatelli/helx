import helx
from helx.rl import baselines
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from bsuite.utils.gym_wrapper import DMEnvFromGym
import wandb


def test_dqn():
    def make(name):
        env = gym.make(name)
        env = RGBImgPartialObsWrapper(env)  # Get pixel observations
        env = ImgObsWrapper(env)  # Get rid of the 'mission' field
        env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
        return env

    env = make("MiniGrid-Empty-6x6-v0")
    hparams = baselines.dqn.HParams(
        replay_memory_size=1000, replay_start=10, batch_size=4
    )
    dqn = baselines.dqn.Dqn(env.observation_spec(), env.action_spec(), hparams)
    wandb.init("dqn", mode="disabled")
    helx.rl.base.run(dqn, env, 100)


if __name__ == "__main__":
    test_dqn()
