import jax
import jax.numpy as jnp
from chex import Shape

from helx.image import greyscale, imresize


def preprocess_atari(x):
    """Preprocessing function from
    Mnih, V., 2015, https://www.nature.com/articles/nature14236
    """
    # depthwise max pooling to remove flickering
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (2, 1, 1, 1), (1, 1, 1, 1), "SAME"
    )
    return greyscale(imresize(x, (84, 84)))


def preprocess_minigrid(x, size: Shape = (56, 56)):
    """Refer to the minigrid implementation at:
    https://github.com/Farama-Foundation/Minigrid
    """
    return imresize(x / 255, size=size, channel_first=False)
