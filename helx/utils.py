import jax
import jax.numpy as jnp


def shuffled_batched_indices(stream_len, batch_size, rng=None, drop_spare=False):
    if rng is None:
        rng = jax.random.PRNGKey(0)
    if isinstance(stream_len, list):
        #  stream_len is a sequence of indices already, or a list of objects
        stream_len = len(stream_len)
    shuffled = jax.random.shuffle(rng, jnp.arange(0, stream_len))
    shuffled_batched = jnp.array_split(
        shuffled,
        jnp.arange(batch_size, stream_len, batch_size),
    )
    if stream_len % batch_size and drop_spare:
        shuffled_batched = shuffled_batched[:-1]
    return shuffled_batched
