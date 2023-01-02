import jax
import jax.numpy as jnp

from chex import Shape


def imresize(
    img, size: Shape, method: str = "bilinear", channel_first: bool = True, **kwargs
):
    assert (
        len(size) == 2
    ), "Size must be a tuple of two numbers, respectively height and width, not {}".format(
        size
    )
    if channel_first:
        size = (*img.shape[:-2], *tuple(size))
    else:
        size = (*img.shape[:-3], *tuple(size), img.shape[-1])
    kwargs.update(shape=size, method=method)
    return jax.image.resize(img, **kwargs)


def greyscale(img):
    """Perceptual luminance-preserving grayscale transformation"""
    luminance_mask = jnp.array([0.2126, 0.7152, 0.0722]).reshape(
        (1,) * (img.ndim - 1) + (3,)
    )
    img = jnp.sum(img * luminance_mask, axis=-1).squeeze()
    return img
