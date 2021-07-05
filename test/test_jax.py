from helx.jax import pure
import os

from helx.nn.module import module
import jax
import jax.numpy as jnp
from jax.experimental.stax import Dense, serial


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


def test_vanilla():
    import jax.numpy as jnp
    import jax
    import jax.experimental.stax as nn
    import jax.experimental.optimizers import adam
    from functools import partial

    jax.config.update("jax_log_compiles", True)

    class Foo:
        def __init__(self, hparams, input_shape, seed=0):
            self.hparams = hparams
            rng = jax.random.PRNGKey(seed)
            init, apply = nn.serial(
                nn.Dense(10),
                nn.Relu,
                nn.Dense(10),
                nn.Relu,
                nn.Dense(1)
            )
            self.params, out_shape = init(rng, input_shape)
            self.optimiser = adam(0.001)

        def loss(self, x, y):
            return jnp.mean(jnp.square(x - y))

        def forward(self, params, x, y):
            y_hat = self.apply(params, x)
            return y_hat, self.loss(y_hat, y)

        @partial(jax.jit, static_argnums=0)
        def step(self, x, y):
            backward = jax.value_and_grad(self.forward, argnums=0)

            error, grads = backward(params, inputs, *args, **kwargs)
            opt_state = optimiser.update(iteration, grads, opt_state)
            return error, opt_state
            backward = jax.grad(self.forward, argnums=1, has_aux=True)
            y_hat, loss = self.forward(self.params, x, y)
            (y_hat, loss), grads = backward()

            return self.loss(y_hat, y)

    #  compiles for sure
    foo = Foo()
    x = jnp.ones((10,))
    y = jnp.ones((10,)) * 2
    foo.step(x, y)

    #  compiles?
    foo.a = 2
    foo.step(x, y)

    #  compiles?
    foo.bla = x
    foo.step(x, y)


if __name__ == "__main__":
    test_pmodule()
    test_pure()
