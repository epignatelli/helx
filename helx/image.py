# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, Shape


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


def greyscale(img: Array, channel_first: bool = False):
    """Perceptual luminance-preserving grayscale transformation"""
    if channel_first:
        shape = (3,) * (img.ndim - 1) + (1,)
    else:
        shape = (1,) * (img.ndim - 1) + (3,)
    luminance_mask = jnp.array([0.2126, 0.7152, 0.0722]).reshape(shape)
    img = jnp.sum(img * luminance_mask, axis=-1).squeeze()
    return img


def ensure_video_format(video: Array, channel_first: bool = True):
    """Ensure that the video has the right shape for `wandb.Video`
    Assumes that the first axis is the time axis.
    Args:
        video (Array): the video to be reshaped.
        channel_first (bool): whether the input array has channel first or not.
    Returns:
        (Array): the reshaped video with shape (time, channel, height, width),
        or None if `video.ndim < 3` or `video.ndim > 5`."""
    # Check shape first
    n_dim = video.ndim
    if n_dim < 3 or n_dim > 5:
        logging.warning(
            "Video must have at least three and at most four dimensions, got {} instead.".format(
                n_dim
            )
        )
        return None

    # video.ndim either 3 or 4
    if n_dim == 3:
        # add channel axis
        video = jnp.expand_dims(video, axis=1)
        video = jnp.repeat(video, repeats=3, axis=1)
    # if 4
    elif not channel_first:
        video = video.transpose(0, 3, 1, 2)

    # Check dtype and range
    if video.dtype == jnp.float32:
        video = jnp.clip(video, 0, 1)
        video = (video * 255).astype(np.uint8)
    return np.array(video)
