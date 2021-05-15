import gym
import jax
from bsuite.utils.gym_wrapper import DMEnvFromGym
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from helx.rl import baselines, environment


def test_dqn():
    def make(name):
        env = gym.make(name)
        env = RGBImgPartialObsWrapper(env)  # Get pixel observations
        env = ImgObsWrapper(env)  #  Get rid of the 'mission' field
        env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
        return env

    hparams = baselines.dqn.HParams(
        replay_memory_size=5000, replay_start=5000, batch_size=32
    )
    env = environment.make_minigrid("MiniGrid-Empty-6x6-v0")
    preprocess = jax.jit(lambda x: x / 255, backend="cpu")
    dqn = baselines.dqn.Dqn(
        (56, 56, 3), env.action_spec().num_values, hparams, preprocess=preprocess
    )


if __name__ == "__main__":
    test_dqn()
