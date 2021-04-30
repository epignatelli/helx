from typing import Sequence
import jax
import jax.numpy as jnp

from ..typing import Value, Return
from .buffer import Trajectory, Transition


def nstep_return(transition: Transition, value: Value) -> Return:
    """n-step return as of:
    Sutton R., Barto. G., 2018, http://incompleteideas.net/book/RLbook2020.pdf
    """
    body_fun = lambda x, i: x * transition.gamma ** i
    n, init = len(transition.r_0), transition.r_0[0]
    return jax.lax.fori_loop(0, n - 1, body_fun, init) + value * transition.gamma ** n


def lambda_return(trajectory: Trajectory, values: Value) -> jnp.ndarray:
    """λ-returns as of:
    Sutton R., Barto. G., 2018, http://incompleteideas.net/book/RLbook2020.pdf
    The off-line λ-return algorithm is a forward view rl algorithm.
    This function returns all the λ-returns for each time step.
    For the backward version see td_lambda.
    This function returns the final λ-return.
    """
    t = trajectory

    def body_fun(i, g):
        return t.rewards[i] + t.gammas[i] * (
            t.lambdas[i] * g + (1 - t.lambdas[i]) * values[i]
        )

    return jax.lax.fori_loop(0, len(t.rewards), body_fun, 0.0)


def lambda_returns(trajectory: Trajectory, values: Value) -> jnp.ndarray:
    """λ-returns as of:
    Sutton R., Barto. G., 2018, http://incompleteideas.net/book/RLbook2020.pdf
    The off-line λ-return algorithm is a forward view rl algorithm.
    For the backward version see td_lambda.
    This function returns all the λ-returns for each time step.
    """
    t = trajectory

    def body_fun(g, i):
        g = t.rewards[i] + t.gammas[i] * (
            t.lambdas[i] * g + (1 - t.lambdas[i]) * values[i]
        )
        return (i + 1, g), g

    return jax.lax.scan(body_fun, (0, 0.0), xs=range(len(t.rewards)))


generalised_advantage = lambda_returns
generalised_advantage.__doc__ = (
    """The Generalised Advantage Estimator (GAE) uses lambda returns.
    See: Schulman, J., 2018, https://arxiv.org/abs/1506.02438\n"""
    + lambda_returns.__doc__
)


def advantage(trajectory: Trajectory, values: Value):
    """Advantage function usused for actor-critic methods as in:
    Mnih, V., 2016, https://arxiv.org/abs/1602.01783
    """


def ewa(series: Sequence, weights: Sequence) -> float:
    """Exponentially weighted average"""
    # makes sure weights and series have same rank
    w = jnp.broadcast_to(weights, series)
    body_fun = lambda x, i: x * w[i] ** i
    return jax.lax.fori_loop(0, len(series), body_fun, (series[0], 0))


def ewa_scan(series: Sequence, weights: Sequence) -> float:
    """Exponentially weighted average that returns
    all the intermediate values"""
    w = jnp.broadcast_to(weights, series)

    def body_fun(carry, x):
        i, g = carry
        y = g + series[i] * (w[i] ** i)
        return y, y

    return jax.lax.scan(body_fun, (0, 0.0), length=len(series))
