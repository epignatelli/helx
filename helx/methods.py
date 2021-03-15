from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import OptimizerState

from .base import factory
from .types import Module, Optimiser, Params


def batch(fun, in_axes=0, out_axes=0, axis_name=None, **jit_kwargs):
    return jax.jit(
        jax.vmap(fun, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name),
        **jit_kwargs
    )


def inject(fun, **kwargs):
    cls = fun.__globals__[fun.__qualname__.split(".")[0]]
    f_jit = jax.jit(fun, **kwargs)
    setattr(cls, fun.__name__, staticmethod(f_jit))
    return inject


def module(fun):
    return factory(fun, Module)


def pure(fun, **kwargs):
    f_jit = jax.jit(fun, **kwargs)

    def wrapper(*a, **k):
        return f_jit(*a, **k)

    return staticmethod(wrapper)


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
