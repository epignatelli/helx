import jax
import jax.numpy as jnp

from ..types import Value, Return
from .buffer import Transition


def nstep_return(transition: Transition, value: Value) -> Return:
    def body_fun(x, i):
        return x * transition.gamma ** i

    n = len(transition.r_1)
    init = transition.r_1[0]
    return jax.lax.fori_loop(0, n, body_fun, init) + value * transition.gamma ** n


def lambda_return(transition: Transition, value: Value) -> jnp.ndarray:
    gamma = transition.gamma
    trace_decay = transition.trace_decay
    rewards = transition.r_1

    def body_fun(carry, x):
        g, (r, v) = carry, x
        g = r + gamma * ((1 - trace_decay) * v + trace_decay) * g
        return g, g

    rewards = jnp.flip(rewards)
    values = jnp.flip(value)
    g = jax.lax.scan(body_fun, (values[0], (rewards, values)))
    return jnp.flip(g)