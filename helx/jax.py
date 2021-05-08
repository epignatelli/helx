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

    #  the actual computation tu run
    def wrapper(*a, **k):
        return f_jit(*a, **k)

    #  staticmethod does not pass the `self` argument when wrapper is called
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


def fori_scan(start, stop, body_fun, init, reversed=False):
    """Uses `jax.lax.scan` to return all the intermediate
    computations of the `jax.lax.fori_loop` function.
    This function is also backward differentiable, unlike `fori_loop`.
    """

    def f(carry, i):
        y = body_fun(i, carry)
        return y, y

    indices = jnp.arange(start, stop)
    indices = jax.lax.cond(
        reversed, lambda _: jnp.flip(indices), lambda _: indices, operand=None
    )
    return jax.lax.scan(f, init, xs=indices)[1]


def device_array(numpy_array, device, *args, **kwargs):
    """Creates a DeviceArray directly on a specified device"""
    return jnp.array(jax.device_put(numpy_array, device), *args, **kwargs)
