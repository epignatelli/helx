import flax.linen as nn
import jax
from chex import Array


@jax.jit
def flatten(x: Array) -> Array:
    return x.reshape((x.shape[0], -1))


class MLP(nn.Module):
    n_layers: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for _ in range(self.n_layers):
            x = nn.Dense(features=32)(x)
            x = nn.relu(x)
        return x


class CNN(nn.Module):
    n_layers: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.n_layers):
            x = nn.Conv(features=i**2, kernel_size=(8, 8), strides=(4, 4))(x)
            x = nn.relu(x)
        x = flatten(x)
        x = nn.Dense(features=128)(x)
        return x
