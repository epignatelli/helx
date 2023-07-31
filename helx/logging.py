import logging
from typing import TypeVar

import wandb

import jax.numpy as jnp
import jax.experimental.host_callback as hcb
import flax.linen as nn

from .agents.agent import Log
from .environment.environment import StepType


T = TypeVar("T", bound=nn.Module)


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
            if log_dict["step_type"] == StepType.TRANSITION:
                continue
        wandb.log({k: v}, commit=False)
    wandb.log({})  # commit flush
    return logs


def host_log_wandb(logs: Log) -> Log:
    hcb.id_tap(lambda x, _: log_wandb(x), logs)
    return logs
