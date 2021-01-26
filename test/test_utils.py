import jax
import jax.numpy as jnp

import helx


def test_shuffled_batch_indices():
    rng = jax.random.PRNGKey(0)
    indices = list(range(10))

    batch_indices_no_drop = list(
        map(
            jnp.array,
            ([2, 7, 9], [6, 0, 8], [1, 3, 4], [5]),
        )
    )
    batch_indices_drop = list(
        map(
            jnp.array,
            ([2, 7, 9], [6, 0, 8], [1, 3, 4]),
        )
    )
    batch_indices_even_drop = list(
        map(
            jnp.array,
            ([2, 7], [9, 6], [0, 8], [1, 3], [4, 5]),
        )
    )

    # test spare, no drop
    res_no_drop = helx.utils.shuffled_batched_indices(
        indices, 3, rng=rng, drop_spare=False
    )
    assert all(
        [
            jnp.array_equal(res_no_drop[i], batch_indices_no_drop[i])
            for i in range(len(batch_indices_no_drop))
        ]
    ), "\n{} \nnot equal to \n{}".format(res_no_drop, batch_indices_no_drop)

    # test spare drop
    res_drop = helx.utils.shuffled_batched_indices(indices, 3, rng=rng, drop_spare=True)
    assert all(
        [
            jnp.array_equal(res_drop[i], batch_indices_drop[i])
            for i in range(len(batch_indices_drop))
        ]
    ), "\n{} \nnot equal to \n{}".format(res_drop, batch_indices_drop)

    # test even drop
    res_even_drop = helx.utils.shuffled_batched_indices(
        indices, 2, rng=rng, drop_spare=True
    )
    assert all(
        [
            jnp.array_equal(res_even_drop[i], batch_indices_even_drop[i])
            for i in range(len(batch_indices_no_drop))
        ]
    ), "\n{} \nnot equal to \n{}".format(res_even_drop, batch_indices_even_drop)
