import logging
from typing import List, Sequence

import dm_env

from .agent import Agent


def run(
    agent: Agent,
    env: dm_env.Environment,
    num_episodes: int,
    eval: bool = False,
) -> Agent:
    logging.info(
        "Starting {} agent {} on environment {}.\nThe scheduled number of episode is {}".format(
            "evaluating" if eval else "training", agent, env, num_episodes
        )
    )
    for episode in range(num_episodes):
        print(
            "Starting episode number {}/{}\t\t\t".format(episode, num_episodes - 1),
            end="\r",
        )
        #  initialise environment
        timestep = env.reset()
        while not timestep.last():
            #  apply policy
            action = agent.policy(timestep)
            #  observe new state
            new_timestep = agent.observe(env, timestep, action)
            #  update policy
            loss = None
            if not eval:
                loss = agent.update(timestep, action, new_timestep)
            #  log update
            agent.log(new_timestep.reward, loss)
            # prepare next iteration
            timestep = new_timestep
    return agent


def run_async(
    agent: Agent,
    env: dm_env.Environment,
    num_episodes: int,
    eval: bool = False,
    n_envs: int = 1,
    seeds: Sequence[int] = None,
) -> Agent:
    """Optimises a policy asynchronously, using `n_envs` in parallel
    to collect new_experience"""
    raise NotImplementedError