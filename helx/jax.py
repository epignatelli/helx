import jax
import jax.numpy as jnp


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

    # set batched version
    cls = fun.__globals__[fun.__qualname__.split(".")[0]]
    setattr(cls, fun.__name__ + "_batch", staticmethod(jax.vmap(fun)))
    return staticmethod(wrapper)


def tree_vmap(f, lst):
    stacked = jax.tree_map(lambda args: jnp.stack(args), lst)
    out_stacked = jax.vmap(f)(stacked)
    _, outer_treedef = jax.tree_flatten([None] * len(lst))
    _, inner_treedef = jax.tree_flatten(out_stacked)
    out_unstacked_transposed = jax.tree_map(list, out_stacked)
    out_unstacked = jax.tree_transpose(
        outer_treedef, inner_treedef, out_unstacked_transposed
    )
    return out_unstacked
