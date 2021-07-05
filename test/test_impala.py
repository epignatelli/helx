import functools
import multiprocessing as mp

import jax
import jax.lib.xla_bridge as xb
import jax.numpy as jnp

import time

from absl import logging

logging.set_verbosity(1)


@functools.partial(jax.jit, static_argnums=0, backend="cpu")
def cpu_ones(shape):
    return jnp.ones(shape)


@functools.partial(jax.jit, backend="gpu")
def gpu_mean(x):
    return jnp.mean(x)


def init_xla():
    xb.backends()


def producer(q, p, i):
    init_xla()
    while True:
        print(
            "[{}]".format(time.time()),
            "[{}]".format(mp.current_process().name),
            "Producing",
            "Process {}".format(i),
            "Size {}".format(q.qsize()),
        )
        x = cpu_ones((10,)) * p
        q.put(x)
        time.sleep(0.5)
    return


def consumer(q, p):
    init_xla()
    while True:
        print(
            "[{}]".format(time.time()),
            "[{}]".format(mp.current_process().name),
            "Consuming",
            "Size {}".format(q.qsize()),
        )
        x = q.get()
        gpu_mean(x) + 1
        time.sleep(1)
    return


def run_async():
    params = jnp.ones((10,))
    context = mp.get_context("spawn")

    m = context.Manager()
    queue = m.Queue(100)
    ps = [
        context.Process(
            target=producer, args=(queue, params, i), name=f"producer-{i}", daemon=True
        )
        for i in range(2)
    ]
    c = context.Process(
        target=consumer, args=(queue, params), name="consumer", daemon=True
    )

    try:
        print("Starting")
        for p in ps:
            p.start()
        c.start()
    except Exception as e:
        print("Failed", repr(e))
    finally:
        p.join()
        c.join()
    return


if __name__ == "__main__":
    run_async()
