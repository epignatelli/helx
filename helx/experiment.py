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


from pprint import pformat

from .agents.agent import Agent
from .environment.base import Environment
from .mdp import Trajectory
from .logging import NullLogger


def run_episode(
    agent: Agent, env: Environment, eval: bool = False, max_steps: int = int(2e9)
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
        action = agent.sample_action(env, eval=eval)
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
    for i in range(num_episodes):

        #  collect experience
        # this can run asynchronously because env.step, env.reset
        # and agent.sample_action might be all non-blocking
        # until the buffer is at minimum learning capacity
        episode = run_episode(agent, env)

        #  update the learner
        # this run asynchronously as well when the
        # agent is a distributed learner and the buffer is full
        log = agent.update(episode)

        #  log the update
        logger.record(log)

        if i % eval_frequency:
            continue

        # evaluate
        # nameof = lambda x: type(x).__name__
        # agent.save("{}-{}.pickle".format(nameof(agent), nameof(env)))

        for j in range(num_eval_episodes):
            log = {}
            logger.log("Evaluating episode {} at iteration {}".format(j, i))
            #  experience a new episode
            episode = run_episode(agent, env, eval=True)
            # video = ensure_video_format(episode.s)
            # if video is not None:
            #     log.update({f"val/$\\pi_{j}$": wandb.Video(video, format="mp4")})

            # log episode
            returns = episode.returns().item()
            log.update({f"val/Return$(\\pi_{j})$": returns})
            logger.log(
                "Episode: {}/{} - Return: {}".format(j, num_eval_episodes - 1, returns)
            )

        # log
        logger.record(log)


# def run_jittable(
#     agent: Agent,
#     env: Environment,
#     num_episodes: int,
#     num_eval_episodes: int = 5,
#     eval_frequency: int = 1000,
#     seed: int = 0
# ):
#     # init logger
#     agent_name = type(agent).__name__
#     env_name = env.name()
#     run_name = "{}/{}/{}".format(agent_name, env_name, seed)
#     log = {}
#     log.update({"run_name": run_name})

#     logging.info(
#         "Starting experiment {}.\nThe scheduled number of episode is {}".format(
#             run_name, num_episodes
#         )
#     )
#     logging.info(
#         "The hyperparameters for the current experiment are {}".format(
#             pformat(agent.hparams.as_dict())
#         )
#     )

#     def eval_fun(carry, x):
#         returns = carry
#         agent, env = x
#         episode = run_episode(agent, env, eval=True)

#         y = (agent, env)
#         return carry, y

#     def train_fun(carry, x):
#         # inputs
#         log = carry
#         agent, env, i = x

#         # train
#         episode = run_episode(agent, env)
#         log = agent.update(episode)

#         # eval
#         if i % eval_frequency:
#             return log, (agent, env)
#         eval_log, _ = jax.lax.scan(eval_fun, log, (agent, env, 0), length=num_eval_episodes)
#         log.update(eval_log)

#         # outputs
#         carry = log
#         y = (agent, env, i + 1)
#         return carry, y

#     # train
#     log, final_state = jax.lax.scan(train_fun, log, (agent, env, 0), length=num_episodes)
#     return log, final_state