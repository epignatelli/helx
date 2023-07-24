from typing import Tuple
import jax
from jax.random import KeyArray
import jax.numpy as jnp
from .environment.environment import Environment

from .mdp import Timestep, StepType
from .agents import Agent
from .logging import Log, host_log_wandb


def run_episode(
    key: KeyArray,
    agent: Agent,
    env: Environment,
    eval: bool = False,
) -> Timestep:
    key, k1 = jax.random.split(key)
    timestep = env.reset(k1)
    timesteps = [timestep]
    while timestep.step_type == StepType.TRANSITION:
        key, k1, k2 = jax.random.split(key, 3)
        action = agent.sample_action(k1, timestep.observation, eval=eval)
        timestep = env.step(k2, timestep, action)
        timesteps.append(timestep)

    # convert list of timesteps into a batched timestep object
    timesteps = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *timesteps)
    return timesteps


def run_n_steps(
    key: KeyArray,
    agent: Agent,
    env: Environment,
    env_state: Timestep,
    n_steps: int,
    eval: bool = False,
) -> Timestep:
    timesteps = []
    for _ in range(n_steps):
        key, k1, k2 = jax.random.split(key, num=3)
        action = agent.sample_action(k1, env_state.observation, eval=eval)
        env_state = env.step(k2, env_state, action)
        timesteps.append(env_state)

    if n_steps == 1:
        return timesteps[0]

    # convert list of timesteps into a batched timestep object
    timesteps = jax.tree_util.tree_map(
        lambda *x: jnp.squeeze(jnp.stack(x), 0), *timesteps
    )
    return timesteps


def run(
    key: KeyArray,
    agent: Agent,
    env: Environment,
    max_timesteps: int,
) -> Tuple[Agent, Timestep]:
    key, k1, k2 = jax.random.split(key, num=3)
    env_state = env.reset(k1)

    def body_fun(
        val: Tuple[Agent, Timestep, Log, KeyArray]
    ) -> Tuple[Agent, Timestep, Log, KeyArray]:
        agent, env_state, log, key = val
        timesteps = run_n_steps(key, agent, env, env_state, n_steps=agent.hparams.n_steps)  # type: ignore
        agent, log = agent.update(key, timesteps, log)
        # potentially blocking, this call is on the host, not on the device, despite jitting
        host_log_wandb(log)
        return agent, env_state, log, key

    agent, env_state, _, _ = jax.lax.while_loop(
        lambda x: x[0].iteration < max_timesteps,
        body_fun,
        (agent, env_state, Log(jnp.asarray(0), jnp.asarray(float('inf')), StepType.TRANSITION, jnp.asarray(0.0)), k2),
    )
    return agent, env_state
