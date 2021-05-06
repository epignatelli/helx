import jax
import jax.numpy as jnp
import threading


def distribute_tree(tree):
    """
    This is usually used to prepare the params for `pmap`ped functions.
    Adds an additional axis to the tree, by duplicating the tree as many times
    as the number returned by `jax.local_device_count()`.
    """
    return jax.tree_map(lambda x: jnp.array([x] * jax.local_device_count()), tree)


def gather_tree(tree):
    """
    This is usually used to reduce back a tree that has been distributed
    using `distribute_tree` or `redistribute_tree`.
    Assumes that the tree is duplicated along the axis 0 an arbitrary number of times,
    and selects only one of the replicas of the pyTree
    """
    return jax.tree_map(lambda x: x[0], tree)


def redistribute_tree(tree):
    """
    This is usually used to prepare the params for `pmap`ped functions when the params
    are coming from a machine with a different number of devices.
    Assumes that the tree is duplicated along the axis 0 an arbitrary number of times.
    The tree usually is a result of a the `distributed_tree` function
    on a machine with a different number of devices.
    This function sets the number of duplicates equal to the
     number returned by `jax.local_device_count()`.
    """
    return distribute_tree(gather_tree(tree))


def distribute_array(array):
    """
    Prepares an array for `jax.pmap` by adding a leading axis
    whose dimension are equal to `jax.local_device_count()`.
    As a results the dimensions of the first axis pre-modification are divided by
    `jax.local_device_count()`. Note that the first dimension needs to be
    divisible by `jax.local_device_count()`.
    """
    n = jax.local_device_count()
    s = array.shape
    assert (
        len(s) > 1 and s[0] % n == 0
    ), "Array dimension {} must be divisible for the number of local devices {}".format(
        s[0], n
    )
    new_shape = (n,) + (s[0] // n,) + s[1:]
    return array.reshape(new_shape)


def gather_array(array):
    """
    Gathers an array that has previously been distributed using `distribute_array`,
    by removing the leading axis and stacking them on the second axis.
    """
    s = array.shape
    assert len(s) > 2, "Array must have at least 3 axes"
    return array.reshape(s[0] + s[1], s[2:])


def redistribute_array(array):
    """
    Gathers and then distributes again a given array.
    See `helx.distributed.distribute_array` and `helx.distributed.gather_array`
    for more
    """
    return distribute_array(gather_array(array))


def async(f):
    """Decorator to run a function asynchronously"""

    def run(*k, **kw):
        t = threading.Thread(target=f, args=k, kwargs=kw, name=f.__name__)
        t.start()
        return

    return run
