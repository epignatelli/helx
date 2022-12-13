from typing import Generator
import jax
import jax.numpy as jnp

from jax.random import KeyArray as Key


KeySequence = Generator[Key, Key, None]


def shuffled_batched_indices(
    rng: Key,
    stream_len: int,
    batch_size: int,
    drop_last: bool = False,
):
    if isinstance(stream_len, list):
        #  stream_len is a sequence of indices already, or a list of objects
        stream_len = len(stream_len)
    shuffled = jax.random.permutation(rng, jnp.arange(0, stream_len))
    shuffled_batched = jnp.array_split(
        shuffled,
        jnp.arange(batch_size, stream_len, batch_size),
    )
    if stream_len % batch_size and drop_last:
        shuffled_batched = shuffled_batched[:-1]
    return shuffled_batched


def PRNGSequence(seed: int) -> KeySequence:
    k = jax.random.PRNGKey(seed)
    while True:
        rng, k = jax.random.split(k)
        yield rng


def new_key(seq: KeySequence) -> Key:
    return next(seq)
