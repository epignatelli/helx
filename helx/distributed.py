import jax
import jax.numpy as jnp


def distribute_tree(tree):
    """
    This is usually used to prepare the params for `pmap`ped functions.
    Adds an additional axis to the tree, by duplicating the tree as many times
    as the number returned by `jax.local_device_count()`.
    """
    return jax.tree_map(lambda x: jnp.array([x] * jax.local_device_count()), tree)


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
    return jax.tree_map(lambda x: jnp.array([x[0]] * jax.local_device_count()), tree)


def gather_tree(tree):
    """
    This is usually used to reduce back a tree that has been distributed
    using `distribute_tree` or `redistribute_tree`.
    Assumes that the tree is duplicated along the axis 0 an arbitrary number of times,
    and selects only one of the replicas of the pyTree
    """
    return jax.tree_map(lambda x: x[0], tree)


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
        len(s) >= 2 and s[0] % n == 0
    ), "Array dimension {} must be divisible for the number of local devices {}".format(
        s[0], n
    )
    new_shape = (n,) + (s[0] // n,) + s[1:]
    return array.reshape(new_shape)
