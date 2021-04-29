import functools
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import OptimizerState

from ..typing import Apply, Init, Params, factory
from ..distributed import distribute_tree
from ..optimise.optimisers import Optimiser


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


def nn(forward_fun, **kwargs):
    @functools.partial(jax.jit, static_argnums=0)
    def backward(
        model: Module, params: Params, *args, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.value_and_grad(forward_fun, argnums=1, has_aux=True)(
            model, params, *args, **kwargs
        )

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def sgd_step(
        model: Module,
        optimiser: Optimiser,
        iteration: int,
        optimiser_state: OptimizerState,
        *args,
        **kwargs
    ) -> Tuple[float, jnp.ndarray, OptimizerState]:
        params = optimiser.params(optimiser_state)
        (loss, y_hat), gradients = backward(model, params, *args, **kwargs)
        return loss, y_hat, optimiser.update(iteration, gradients, optimiser_state)

    cls = forward_fun.__globals__[forward_fun.__qualname__.split(".")[0]]
    forward_jit = jax.jit(forward_fun, **kwargs)
    setattr(cls, forward_fun.__name__, staticmethod(forward_jit))
    setattr(cls, backward.__name__, staticmethod(backward))
    setattr(cls, sgd_step.__name__, staticmethod(sgd_step))
    return nn
