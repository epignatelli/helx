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
        env = ImgObsWrapper(env)  #  Get rid of the 'mission' field
        env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
        return env

    env = make("MiniGrid-Empty-6x6-v0")
    hparams = baselines.dqn.HParams(
        replay_memory_size=5000, replay_start=5000, batch_size=32
    )
    dqn = baselines.dqn.Dqn((56, 56, 3), env.action_spec().num_values, hparams)
    dqn.log = lambda x, y, z, u, v: None
    helx.rl.run.run(dqn, env, 1000000)


if __name__ == "__main__":
    test_dqn()
