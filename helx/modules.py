import jax

from .types import Module
from .base import factory


def inject(fun, **kwargs):
    cls = fun.__globals__[fun.__qualname__.split(".")[0]]
    f_jit = jax.jit(fun, **kwargs)
    setattr(cls, fun.__name__, staticmethod(f_jit))
    return inject


def module(fun):
    return factory(fun, Module)


def batch(fun, *vmap_args, **jit_kwargs):
    return jax.jit(jax.vmap(fun, *vmap_args), **jit_kwargs)