# Copyright 2023 The Helx Authors.
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
from __future__ import annotations
from functools import partial

from typing import Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array


class Split(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, ...]:
        return (x,) * self.n_heads


class Merge(nn.Module):
    aggregate: Callable[[Array], Array] = partial(jnp.sum, axis=-1)

    @nn.compact
    def __call__(self, *arrays: Tuple[Array, ...]) -> Array:
        x = jnp.broadcast_arrays(*arrays)  # type: ignore
        x = jnp.asarray(x)
        return self.aggregate(x)


class Parallel(nn.Module):
    functions: Tuple[nn.Module, ...]

    @nn.compact
    def __call__(self, *inputs: Tuple[Array, ...]) -> Tuple[Array, ...]:
        return jtu.tree_map(lambda f, x: f(x), self.functions, inputs)


class Lambda(nn.Module):
    function: Callable[[Array], Array]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return self.function(x)


class Flatten(nn.Module):
    """A Flax module that flattens the input array."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x.flatten()


class Identity(nn.Module):
    """A Flax module that returns the input array."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> Array:
        temperature = self.param(
            "temperature_value",
            init_fn=lambda key: jnp.log(self.initial_temperature),
        )
        return jnp.exp(temperature)


class MLP(nn.Module):
    """Defines a multi-layer perceptron by alternating a dense layer with a ReLU
    non-linearity.
    Args:
        features (Sequence[int]): A sequence of integers that define the number of
            units in each dense layer. E.g., (32, 64) defines a network with two
            dense layers, the first with 32 units and the second with 64.
    Example:
        >>> import jax.numpy as jnp
        >>> from helx.networks import MLP
        >>> mlp = MLP(features=(32, 64))
        >>> x = jnp.ones((1, 28, 28))
        >>> mlp(x).shape
        (1, 64)"""

    features: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, x: Array, *args, **kwargs) -> Array:
        for i in range(len(self.features)):
            x = nn.Dense(features=self.features[i])(x)
            x = self.activation(x)
        return x


class CNN(nn.Module):
    """Defines a convolutional neural network by alternating a
    convolution with a ReLU non-linearity.
    There is no pooling or other non-linearities between convolutions.
    Args:
        features (Tuple[int, ...]): A sequence of integers that define the number of
            filters in each convolution. E.g., (32, 64) defines a network with
            two convolutions, the first with 32 filters and the second with 64.
        kernel_sizes (Tuple[Tuple[int, ...]]): A sequence of tuples that define
            the kernel size of each convolution.
        strides (Tuple[Tuple[int, ...]]): A sequence of tuples that define the
            stride of each convolution.
        padding (Tuple[nn.linear.PaddingLike]): A sequence of padding modes for
            each convolution.

    Example:
        >>> import jax.numpy as jnp
        >>> from helx.networks import CNN
        >>> cnn = CNN(
        ...     features=(32, 64),
        ...     kernel_sizes=((3, 3), (3, 3)),
        ...     strides=((1, 1), (1, 1)),
        ...     padding=("SAME", "SAME"),
        ... )
        >>> cnn(jnp.ones((1, 10, 10, 3))).shape
        (1, 10, 10, 64)
    """

    features: Tuple[int, ...] = (32,)
    kernel_sizes: Tuple[Tuple[int, ...], ...] = ((3, 3),)
    strides: Tuple[Tuple[int, ...], ...] = ((1, 1),)
    paddings: Tuple[nn.linear.PaddingLike, ...] = ("SAME",)
    activation: Callable[[Array], Array] = nn.relu
    flatten: bool = False

    def setup(self):
        assert (
            len(self.features)
            == len(self.kernel_sizes)
            == len(self.strides)
            == len(self.paddings)
        ), "All arguments must have the same length, got {} {} {} {}".format(
            len(self.features),
            len(self.kernel_sizes),
            len(self.strides),
            len(self.paddings),
        )
        self.modules = [
            nn.Conv(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                strides=self.strides[i],
                padding=self.paddings[i],
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        for module in self.modules:
            x = module(x)
            x = self.activation(x)
        if self.flatten:
            x = x.flatten()
        return x
