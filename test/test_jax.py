from helx.jax import pure
import os

from helx.nn.module import module
import jax
import jax.numpy as jnp
from jax.experimental.stax import Dense, serial
import jax.experimental.stax as nn
from jax.experimental.optimizers import adam
from functools import partial

jax.config.update("jax_log_compiles", True)


def test_pmodule():
    jax.config.update("jax_platform_name", "cpu")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    n_devices = jax.local_device_count()
    print(n_devices)

    @module
    def m(a, b):
        return serial(Dense(a), Dense(b))

    m(2, 3)


def test_pure():
    class Foo:
        @pure
        def impure(x):
            return jnp.square(x)

    foo = Foo()
    foo.impure(jnp.ones((10, 10)))
    jax.jit(jax.grad(foo.impure))(1.0)


if __name__ == "__main__":
    test_pmodule()
    test_pure()
