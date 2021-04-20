from functools import partial
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import OptimizerState

from .base import factory
from .distributed import distribute_tree
from .types import Module, Optimiser, Params, Scheduler


def batch(fun, in_axes=0, out_axes=0, axis_name=None, **jit_kwargs):
    """
    A utility wrapper around `jax.vmap` + `jax.jit`.
    """
    return jax.jit(
        jax.vmap(fun, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name),
        **jit_kwargs
    )


def inject(fun, **kwargs):
    """
    It is a common pattern to define closures of classes with JAX.
    `inject` adds the closure to the outer class as a static method, and jits it,
    The function after `inject` is pure, and can be used as any other function in JAX.
    """
    cls = fun.__globals__[fun.__qualname__.split(".")[0]]
    f_jit = jax.jit(fun, **kwargs)
    setattr(cls, fun.__name__, staticmethod(f_jit))
    return inject


def pure(fun, **kwargs):
    """
    Puryfies a class function to be used as any jax function, and finally jits it.
    """
    f_jit = jax.jit(fun, **kwargs)

    def wrapper(*a, **k):
        return f_jit(*a, **k)

    return staticmethod(wrapper)


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


def scheduler(fun):
    return factory(fun, Scheduler)


def nn(forward_fun, **kwargs):
    @partial(jax.jit, static_argnums=0)
    def backward(
        model: Module, params: Params, *args, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.value_and_grad(forward_fun, argnums=1, has_aux=True)(
            model, params, *args, **kwargs
        )

    @partial(jax.jit, static_argnums=(0, 1))
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
