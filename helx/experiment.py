import logging

import dm_env
import jax.numpy as jnp
import numpy as np

import wandb

from .agents.agent import Agent


def run(
    agent: Agent,
    env: dm_env.Environment,
    num_episodes: int,
    num_eval_episodes: int = 5,
    eval_frequency: int = 1000,
    print_frequency=10,
    project="",
    experiment_name: str = "",
    debug: bool = False
):
    # init logger
    nameof = lambda x: type(x).__name__
    env_name, agent_name = list(map(nameof, (env, agent)))
    mode = ("online", "disabled")[int(debug)]
    wandb.init(
        project=project,
        group="{}-{}".format(env_name, agent_name),
        tags=(env_name, agent_name, experiment_name),
        config=agent.hparams._asdict(),
        mode=mode
    )

    logging.info(
        "Starting {} agent {} on environment {}.\nThe scheduled number of episode is {}".format(
            "evaluating" if eval else "training", agent_name, env_name, num_episodes
        )
    )
    logging.info(
        "The hyperparameters for the current experiment are {}".format(
            agent.hparams._asdict()
        )
    )
    logging.info("Start logging experiment on wandb project {}".format(experiment_name))

    # train
    for i in range(num_episodes):

        #  experience a new episode
        episode = agent.unroll(env)
        returns = jnp.sum(episode.r).item()

        #  update policy
        loss = agent.update(episode)

        # log episode
        if i % print_frequency == 0:
            logging.info(
                "Iteration: {}/{} - Return: {} - Loss: {}".format(
                    agent.iteration, num_episodes - 1, returns, loss
                )
            )

        if i % eval_frequency:
            continue

        # evaluate
        # nameof = lambda x: type(x).__name__
        # agent.save("{}-{}.pickle".format(nameof(agent), nameof(env)))

        expected_returns = 0.0
        for j in range(num_eval_episodes):
            logging.info("Evaluating episode {} at iteration {}".format(j, i))
            #  experience a new episode
            episode = agent.unroll(env, eval=True)
            returns = jnp.sum(episode.r).item()
            video = np.array(episode.s).transpose((0, 3, 1, 2))
            wandb.log({f"val/policy-{j}": wandb.Video(video, format="mp4")})

            # log episode
            logging.info(
                "Episode: {}/{} - Return: {}".format(j, num_eval_episodes - 1, returns)
            )
            expected_returns += returns

        expected_returns /= num_eval_episodes
        wandb.log({"val/Return": expected_returns})
