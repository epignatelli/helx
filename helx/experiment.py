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


import logging
from pprint import pformat

import wandb

from .agents.agent import Agent
from .environment.base import Environment
from .mdp import Episode
from .image import ensure_video_format


def run_episode(
    agent: Agent, env: Environment, eval: bool = False, max_steps: int = int(2e9)
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
    while (not timestep.is_final()) and t < max_steps:
        t += 1
        action = agent.sample_action(timestep.observation, eval=eval)
        timestep = env.step(action)
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
    experiment_name: str = "debug",
    debug: bool = False,
):
    # init logger
    agent_name = type(agent).__name__
    env_name = env.name()
    run_name = "{}/{}/{}".format(experiment_name, agent_name, env_name)
    mode = ("online", "disabled")[int(debug)]
    wandb.init(
        project=project,
        group="{}/{}".format(env_name, agent_name),
        tags=[env_name, agent_name, experiment_name],
        config=agent.hparams.as_dict(),
        name=run_name,
        mode=mode,
    )

    logging.info(
        "Starting experiment {}.\nThe scheduled number of episode is {}".format(
            run_name, num_episodes
        )
    )
    logging.info(
        "The hyperparameters for the current experiment are {}".format(
            pformat(agent.hparams.as_dict())
        )
    )
    logging.info("Start logging experiment on wandb project {}".format(experiment_name))

    # train
    for i in range(num_episodes):

        #  experience a new episode
        episode = run_episode(agent, env)
        #  update policy
        loss = agent.update(episode)

        # log episode
        returns = episode.returns().item()
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
            video = ensure_video_format(episode.s)
            if video is not None:
                wandb.log({f"val/policy-{j}": wandb.Video(video, format="mp4")})

            # log episode
            returns = episode.returns().item()
            logging.info(
                "Episode: {}/{} - Return: {}".format(j, num_eval_episodes - 1, returns)
            )
            expected_returns += returns

        expected_returns /= num_eval_episodes
        wandb.log({"val/Return": expected_returns})
