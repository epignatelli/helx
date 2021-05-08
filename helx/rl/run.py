import logging

import dm_env

from .agent import IAgent


def run(
    agent: IAgent,
    env: dm_env.Environment,
    num_episodes: int,
    eval: bool = False,
) -> IAgent:
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
            env, new_timestep = agent.observe(env, timestep, action)
            #  update policy
            loss = None
            if not eval:
                loss = agent.update(timestep, action, new_timestep)
            #  log update
            agent.log(timestep, action, new_timestep, loss)
            # prepare next iteration
            timestep = new_timestep
    return agent
