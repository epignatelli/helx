import jax

from .types import Module
from .base import factory


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


def purify(fun, **kwargs):
    f_jit = jax.jit(fun, **kwargs)

    def wrapper(*a, **k):
        return f_jit(*a, **k)

    return staticmethod(wrapper)
