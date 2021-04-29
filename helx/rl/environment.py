import jax
from jax.api import grad
import jax.numpy as jnp
from bsuite.utils.gym_wrapper import DMEnvFromGym
from gym_minigrid.wrappers import *

from ..typing import Size
from ..image import greyscale, imresize


def make(name):
    env = gym.make(name)
    env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
    return env


def make_minigrid(name):
    env = gym.make(name)
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
    return env


def preprocess_atari(x):
    """Preprocessing function from
    Mnih, V., 2015, https://www.nature.com/articles/nature14236
    """
    # depthwise max pooling to remove flickering
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (2, 1, 1, 1), (1, 1, 1, 1), "SAME"
    )
    # grayscale
    y = greyscale(x)
    # resize, x is (n_frames, 210, 160)
    y = imresize(y, 84, 84)
    return y


def preprocess_minigrid(x, size: Size = (56, 56)):
    """Refer to the minigrid implementation at:
    https://github.com/maximecb/gym-minigrid
    """
    return greyscale(imresize(x, size)
