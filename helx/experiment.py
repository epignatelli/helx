import logging

import dm_env
import jax.numpy as jnp
import numpy as np

import wandb

from .agents.agent import Agent
from .environment.environment import Environment
from .environment.mdp import Episode


def run_episode(
    agent, env: Environment, eval: bool = False, max_steps: int = int(2e9)
) -> Episode:
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
    episode = Episode.start(timestep)
    while (not timestep.last()) and t < max_steps:
        t += 1
        action, _ = agent.sample_action(timestep.observation, eval=eval)
        timestep = env.step(action.item())
        episode.add(timestep, action)
    return episode



def run(
    agent: Agent,
    env: Environment,
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
        episode = run_episode(agent, env)
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
            episode = run_episode(agent, env, eval=True)
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
