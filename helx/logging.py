import logging
from pprint import pformat
from typing import TypeVar
import logging

import wandb

import jax.numpy as jnp
import jax.experimental.host_callback as hcb
import flax.linen as nn

from .agents.agent import Log
from .mdp import TRANSITION


T = TypeVar("T", bound=nn.Module)

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    WHITE = '\033[1;37m'


def log_wandb(logs: Log) -> Log:
    log_dict = logs.__dict__
    for k, v in log_dict.items():
        if k.startswith("_"):
            continue
        if jnp.isnan(v):
            logging.warning(f"NaN value for {k}")
        # do not log returns unless episode has finished
        if k == "returns":
            if not "step_type" in log_dict:
                raise ValueError("Log must have step_type to log returns")
            if log_dict["step_type"] == TRANSITION:
                continue
        wandb.log({k: v}, commit=False)
    wandb.log({})  # commit flush
    return logs


def host_log_wandb(logs: Log) -> Log:
    hcb.id_tap(lambda x, _: log_wandb(x), logs)
    return logs


def log_start(seed, agent, env, budget):
    logging.info(Color.BOLD + Color.WHITE + "Experiment starts with seed {} and budget {}".format(seed, budget) + Color.RESET)
    logging.info(pformat(agent))
    logging.info(pformat(env))


def log_end(seed, agent, env, budget):
    logging.info(Color.BOLD + Color.WHITE + "Experiment completed with seed {} and budget {}".format(seed, budget) + Color.RESET)
    logging.info(pformat(agent))
    logging.info(pformat(env))


def report(run_fn):
    def wrapped(seed, agent, env, budget):
        log_start(seed, agent, env, budget)
        results = run_fn(seed, agent, env, budget)
        log_end(seed, agent, env, budget)
        return results
    return wrapped
