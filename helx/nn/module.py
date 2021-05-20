import functools
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from ..typing import Apply, Init, factory
from ..distributed import distribute_tree


InitState = Callable[[], jnp.ndarray]


class Module(NamedTuple):
    init: Init
    apply: Apply


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
