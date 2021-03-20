import jax
import jax.numpy as jnp


def distribute_tree(tree):
    """
    This is usually used to prepare the params for `pmap`ped functions.
    Adds an additional axis to the tree, by duplicating the tree as many times
    as the number returned by `jax.local_device_count()`.
    """
    return jax.tree_map(lambda x: jnp.array([x] * jax.local_device_count()), tree)


def reistribute_tree(tree):
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