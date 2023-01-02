from typing import Sequence, Tuple
import flax.linen as nn
import jax
from chex import Array
from flax.core.scope import FrozenVariableDict


class Flatten(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x.flatten()


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i in range(len(self.features)):
            x = nn.Dense(features=self.features[i])(x)
            x = nn.relu(x)
        return x


class CNN(nn.Module):
    features: Tuple[int] = (32,)
    kernel_sizes: Tuple[Tuple[int, ...]] = ((3, 3),)
    strides: Tuple[Tuple[int, ...]] = ((1, 1),)
    padding: Tuple[nn.linear.PaddingLike] = ("SAME",)

    def setup(self):
        assert (
            len(self.features)
            == len(self.kernel_sizes)
            == len(self.strides)
            == len(self.padding)
        ), "All arguments must have the same length, got {} {} {} {}".format(
            len(self.features),
            len(self.kernel_sizes),
            len(self.strides),
            len(self.padding),
        )
        self.modules = [
            nn.Conv(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                strides=self.strides[i],
                padding=self.padding[i],
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
            x = nn.relu(x)
        return x
