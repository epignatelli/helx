import functools
from typing import Callable, NamedTuple

from jax.experimental.stax import serial, elementwise
import jax.numpy as jnp
from jax import lax


from helx.typing import Apply, Init, factory
from helx.distributed import distribute_tree


InitState = Callable[[], jnp.ndarray]


operator = lambda op, y: elementwise(functools.partial(op, y=y))


class Module(NamedTuple):
    init: Init
    apply: Apply
    
    def __call__(self, y):
        if not isinstance(y, Module):
            try:
                y = Module(*y)
            except:
                raise ValueError("y must be an instance of type {}, got {} instead".format(Module, type(y)))
        return serial(self, y)
    
    def __add__(self, y):
        return serial(self, operator(lax.add, y))

    def __sum__(self, y):
        return serial(self, operator(lax.sub, y))

    def __mul__(self, y: jnp.ndarray):
        return serial(self, operator(lax.mul, y))

    def __floordiv__(self, y: jnp.ndarray):
        return serial(self, operator(lax.div, y))

    def __truediv__(self, y: jnp.ndarray):
        return serial(self, operator(lax.div, y))

    def __matmul__(self, y: jnp.ndarray):
        return serial(self, elementwise(functools.partial(lax.batch_matmul, rhs=y)))

    def __pow__(self, y: jnp.ndarray):
        return serial(self, operator(lax.pow, y))

    def __eq__(self, y: object):
        return serial(self, operator(lax.eq, y))

    def __ne__(self, y: object):
        return serial(self, operator(lax.ne, y))

    def __gt__(self, y: object):
        return serial(self, operator(lax.gt, y))

    def __ge__(self, y: object):
        return serial(self, operator(lax.ge, y))

    def __lt__(self, y: object):
        return serial(self, operator(lax.lt, y))

    def __le__(self, y: object):
        return serial(self, operator(lax.le, y))


def module(fun):
    """
    Decorator method to define `Module`s.
    The decorator wraps the desired function and returns a Module object.
    """
    return factory(fun, Module)


def pmodule(fun):
    """
    Same as `helx.methods.module` with an additional axis for the parameters
    to be used with `jax.pmap`.
    """

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        module = Module(*fun(*args, **kwargs))

        def init(input_shape, rng):
            output_shape, params = module.init(input_shape, rng)
            params = distribute_tree(params)
            return output_shape, params

        return module._replace(init=init)

    return wrapper
