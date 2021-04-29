import jax
import jax.numpy as jnp

from .typing import Size


def imresize(img, size: Size, **kwargs):
    assert (
        len(size) == 2
    ), "Size must be a tuple of two numbers, respectively height and width, not {}".format(
        size
    )
    size = (*img.shape[:-3], *tuple(size))
    return jax.image.resize(img, **kwargs.update(size=size))


def greyscale(img):
    """Perceptual luminance-preserving grayscale transformation"""
    luminance_mask = jnp.array([0.2126, 0.7152, 0.0722]).reshape(1, 1, 1, 3)
    img = jnp.sum(img * luminance_mask, axis=-1).squeeze()
    return img
