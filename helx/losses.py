from typing import Tuple
import flax.linen as nn
from flax.core.scope import VariableDict as Params
from jax import Array
import jax.numpy as jnp
import optax
import rlax

from .mdp import TERMINATION, Timestep


def flatten_timesteps(timesteps: Timestep, discount: float = 1.0) -> Tuple[Array, Array, Array, Array, Array, Array]:
    s_tm1 = timesteps.observation[:-1]
    s_t = timesteps.observation[1:]
    a_tm1 = timesteps.action[:-1][0]  # [0] because scalar
    r_t = timesteps.reward[:-1][0]  # [0] because scalar
    terminal_tm1 = timesteps.step_type[:-1] != TERMINATION
    discount_t = discount ** timesteps.t[:-1][0]  # [0] because scalar
    return s_tm1, s_t, a_tm1, r_t, terminal_tm1, discount_t

def dqn_loss(
    timesteps: Timestep,
    critic: nn.Module,
    params: Params,
    params_target: Params,
    discount: float = 0.99,
) -> Array:
    s_tm1, s_t, a_tm1, r_t, terminal_tm1, discount_t = flatten_timesteps(timesteps, discount)

    q_tm1 = jnp.asarray(critic.apply(params, s_tm1))
    q_t = jnp.asarray(critic.apply(params_target, s_t)) * terminal_tm1

    td_error = rlax.q_learning(
        q_tm1, a_tm1, r_t, discount_t, q_t, stop_target_gradients=True
    )
    td_loss = jnp.mean(0.5 * td_error**2)
    return td_loss


def double_dqn_loss(
    timesteps: Timestep,
    critic: nn.Module,
    params: Params,
    params_target: Params,
    discount: float = 0.99,
):
    s_tm1, s_t, a_tm1, r_t, terminal_tm1, discount_t = flatten_timesteps(timesteps, discount)

    q_tm1 = jnp.asarray(critic.apply(params, s_tm1))
    a_t = jnp.argmax(jnp.asarray(critic.apply(params, s_t)) * terminal_tm1)
    q_t = critic.apply(params_target, s_t)
    q_target = r_t + discount_t * q_t[a_t]

    td_loss = optax.l2_loss(q_tm1[a_tm1] - q_target)
    return jnp.asarray(td_loss)


def soft_q_loss(
    timesteps: Timestep,
    critic: nn.Module,
    params: Params,
    params_target: Params,
    entropy: Array,
    discount: float = 0.99,
):
    s_tm1, s_t, a_tm1, r_t, terminal_tm1, discount_t = flatten_timesteps(timesteps, discount)


    raise NotImplementedError()
