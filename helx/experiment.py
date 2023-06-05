# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial
from pprint import pformat

import jax
import jax.experimental
from jax.random import KeyArray

from .agents.agent import Agent
from .environment.base import Environment
from .mdp import Trajectory
from .logging import NullLogger


def run_episode(
    agent: Agent, env: Environment, key: KeyArray, eval: bool = False, max_steps: int = int(2e9)
) -> Trajectory:
    """Deploys the agent in the environment for a full episode.
        In case of batched environments, the each property of the episode
        has an additional `batch` axis at index `0`.
        The episode terminates only for absorbing states, and the number of
        maximum steps can be specified in the environment itself.
        While using `experiment.run` the result of this function is passed
        to the `update` method.
    Args:
        env (dm_env.Environment): the environment to interact with implements
            the `dm_env.Environment` interface, and NOT a `gym` interface.
            You can wrap the former around the latter using  the
            `bsuite.utils.gym_wrapper.DMFromGymWrapper` wrapper.
        eval (bool): eval flag passed to the `policy` method.
    Returns:
        (Episode): an full episode following the current policy.
            Each property of the episode has an additional `batch` axis at index `0`.
            in case of batched environments.
    """
    t = 0
    timestep = env.reset()
    episode = Trajectory.start(timestep)
    while (not timestep.is_final()) and t < max_steps:
        t += 1
        action = agent.sample_action(env, key=key, eval=eval)
        timestep = env.step(action)
        episode.add(timestep, action)
    return episode


def run(
    agent: Agent,
    env: Environment,
    num_episodes: int,
    num_eval_episodes: int = 5,
    eval_frequency: int = 1000,
    logger=NullLogger(),
    seed: int=0
):
    logger.log(
        "Starting experiment {} with a budget of {} episodes".format(
            logger.experiment_name, num_episodes
        )
    )
    logger.log(
        "The hyperparameters for the current experiment are {}".format(
            pformat(agent.hparams.as_dict())
        )
    )

    # train
    log = {}
    key = jax.random.PRNGKey(seed)
    for i in range(num_episodes):
        key, k1, k2 = jax.random.split(key, 3)
        # collect experience
        # this can run asynchronously because env.step, env.reset
        # and agent.sample_action might be all non-blocking
        # until the buffer is at minimum learning capacity
        episode = run_episode(agent=agent, env=env, key=k1)
        returns = episode.returns().item()

        # log episode
        logger.log(f"Learning episode {i}/{num_episodes} - Return: {returns}")
        log.update({f"train/Returns": returns})

        # update the learner
        # this run asynchronously as well when the
        # agent is a distributed learner and the buffer is full
        agent, log = agent.update(episode=episode, key=k2)

        # log the update
        logger.record(log)

        if i % eval_frequency:
            continue

        # evaluate
        # nameof = lambda x: type(x).__name__
        # agent.save("{}-{}.pickle".format(nameof(agent), nameof(env)))

        for j in range(num_eval_episodes):
            log = {}

            #  create new rng key
            key, k3 = jax.random.split(key)

            #  experience a new episode
            episode = run_episode(agent, env, key=k3, eval=True)
            returns = episode.returns().item()

            # log episode
            logger.log(f"Evaluating episode {j}/{num_eval_episodes} - Return: {returns}")
            log.update({f"val/Return": returns})

        # log
        logger.record(log)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def run_jitted(
    agent: Agent,
    env: Environment,
    num_episodes: int,
    num_eval_episodes: int = 5,
    eval_frequency: int = 1000,
    logger=NullLogger(),
    seed=0
):
    logger.log(
        "Starting experiment {} with a budget of {} episodes".format(
            logger.experiment_name, num_episodes
        )
    )
    logger.log(
        "The hyperparameters for the current experiment are {}".format(
            pformat(agent.hparams.as_dict())
        )
    )

    def eval_fun(carry, log):
        agent, key, i = carry

        key, k3 = jax.random.split(key, 2)

        episode = run_episode(agent, env, key=k3, eval=True)
        log = {f"val/Return$(\\pi_{i})$": episode.returns()}

        # log eval
        jax.debug.print(pformat(log))
        jax.experimental.io_callback(logger.record, (), log, False)

        carry = agent, key, i + 1
        return carry, log

    def train_fun(carry, log):
        agent, key, i = carry

        key, k1, k2 = jax.random.split(key, 3)

        # train
        episode = run_episode(agent=agent, env=env, key=k1)
        agent, log = agent.update(episode, key=k2)

        # log train
        jax.debug.print(pformat(log))
        jax.experimental.io_callback(logger.record, (), log, False)

        # eval
        if i % eval_frequency:
            return log, (agent, env)
        init_eval_carry = (agent, 0)
        _, eval_log = jax.lax.scan(eval_fun, init_eval_carry, None, length=num_eval_episodes)
        log.update(eval_log)

        carry = agent, key, i + 1
        return carry, log

    # train
    key = jax.random.PRNGKey(seed)
    init_carry = (agent, key, 0)
    carry, log = jax.lax.scan(train_fun, init_carry, None, length=num_episodes)
    agent, key, i = carry
    return log, agent
