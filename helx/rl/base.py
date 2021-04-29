import abc
import logging
from typing import Tuple

import dm_env
import jax
from bsuite.baselines import base
from helx.rl.buffer import Transition
from jax.experimental.optimizers import OptimizerState

from ..jax import pure
from ..nn.module import Module
from ..optimise.optimisers import Optimiser
from ..typing import HParams, Loss, Params


class Agent(base.Agent):
    network: Module
    optimiser: Optimiser
    hparams: HParams

    def __init__(self, network, optimiser, hparams):
        Agent.network = network
        Agent.optimiser = optimiser
        Agent.hparams = hparams
        self._iteration = 0

    @abc.abstractmethod
    def loss(params: Params, transition: Transition) -> Loss:
        """Specifies the loss function to be minimised by some flavour of SGD"""

    @pure
    def sgd_step(
        iteration: int,
        opt_state: OptimizerState,
        transition: Transition,
    ) -> Tuple[Loss, OptimizerState]:
        """Performs a gradient descent step"""
        params = Agent.optimiser.params(opt_state)
        backward = jax.value_and_grad(Agent.loss, argnums=2)
        error, grads = backward(Agent.network, params, transition)
        return error, Agent.optimiser.update(iteration, grads, opt_state)


def run(
    agent: Agent,
    env: dm_env.Environment,
    num_episodes: int,
    eval_mode: bool = False,
) -> Agent:
    logging.info(
        "Starting {} agent {} on environment {}.\nThe scheduled number of episode is {}".format(
            "evaluating" if eval_mode else "training", agent, env, num_episodes
        )
    )
    for episode in range(num_episodes):
        print(
            "Starting episode number {}/{}\t\t\t".format(episode, num_episodes - 1),
            end="\r",
        )
        # initialise environment
        timestep = env.reset()
        while not timestep.last():
            # policy
            action = agent.select_action(timestep)
            # step environment
            new_timestep = env.step(action)
            # update
            loss = None
            if not eval_mode:
                loss = agent.update(timestep, action, new_timestep)
            agent.log(new_timestep.reward, loss)
            # prepare next
            timestep = new_timestep
    return agent
