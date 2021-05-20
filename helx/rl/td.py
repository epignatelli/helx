from functools import partial
from typing import Sequence
import jax
import jax.numpy as jnp

from helx.typing import Value, Return
from helx.jax import fori_scan
from .memory import Trajectory


def nstep_return(trajectory: Trajectory, value: Value) -> Return:
    """n-step return as of:
    Sutton R., Barto. G., 2018, http://incompleteideas.net/book/RLbook2020.pdf
    """
    body_fun = (
        lambda i, g: trajectory.rewards[-i - 1] + trajectory.discounts[-i - 1] * g
    )
    n, init = len(trajectory.rewards), jnp.zeros_like(trajectory.rewards)
    return (
        jax.lax.fori_loop(0, n - 1, body_fun, init)
        + value * trajectory.discounts[0] ** n
    )


def nstep_returns(trajectory: Trajectory, value: Value) -> Return:
    """n-step return as of:
    Sutton R., Barto. G., 2018, http://incompleteideas.net/book/RLbook2020.pdf
    """
    body_fun = lambda i, x: x * trajectory.discounts[i] ** i
    n, init = len(trajectory.rewards), trajectory.rewards[-1] * 0.0
    return (
        fori_scan(body_fun, 0, n - 1, body_fun, init)
        + value * trajectory.discounts[-1] ** n
    )


def _lambda_return_t(t, v, i, g):
    return t.rewards[i] + t.discounts[i] * (
        t.trace_decays[i] * g + (1 - t.trace_decays[i]) * v[i]
    )


def lambda_return(trajectory: Trajectory, values: Value) -> jnp.ndarray:
    """λ-returns as of:
    Sutton R., Barto. G., 2018, http://incompleteideas.net/book/RLbook2020.pdf
    The off-line λ-return algorithm is a forward view rl algorithm.
    This function returns all the λ-returns for each time step.
    For the backward version see td_lambda.
    This function returns the final λ-return.
    """
    body_fun = partial(_lambda_return_t, t=trajectory, v=values)
    return jax.lax.fori_loop(0, len(trajectory.rewards), body_fun, 0.0)


def lambda_returns(trajectory: Trajectory, values: Value) -> jnp.ndarray:
    body_fun = (
        lambda g, i: (partial(_lambda_return_t, t=trajectory, v=values)(i, g),) * 2
    )
    # the recursive formulation of λ-returns requires to iterate backwards
    return jnp.flip(fori_scan(0, len(trajectory.rewards), body_fun, 0.0, reversed=True))


lambda_returns.__doc__ = (
    lambda_return.__doc__
    + """This function returns all the λ-returns for each time step.
"""
)

generalised_advantage = lambda_returns
generalised_advantage.__doc__ = (
    """The Generalised Advantage Estimator (GAE) uses truncated lambda returns.
    See: Schulman, J. et al., 2018, https://arxiv.org/abs/1506.02438\n"""
    + lambda_returns.__doc__
)


def advantages(trajectory: Trajectory, v_t: Value, v_tk: Value):
    """Advantage function used for actor-critic methods as in:
    Mnih, V. et al., 2016, https://arxiv.org/abs/1602.01783
    """
    n = len(trajectory.rewards)
    g_t = ewas(trajectory.rewards, trajectory.gammas)
    return g_t + trajectory.gammas[-1] ** n * v_tk - v_t


def ewa(series: Sequence, weights: Sequence) -> float:
    """Exponentially weighted average"""
    # makes sure weights and series have same rank
    w = jnp.broadcast_to(weights, series)
    body_fun = lambda x, i: x * w[i] ** i
    return jax.lax.fori_loop(0, len(series), body_fun, (series[0], 0))


def ewas(series: Sequence, weights: Sequence) -> jnp.ndarray:
    """Exponentially weighted average that returns
    all the intermediate values"""
    w = jnp.broadcast_to(weights, series)
    body_fun = lambda x, i: x * w[i] ** i
    return fori_scan(0, len(series), body_fun, (series[0], 0))
