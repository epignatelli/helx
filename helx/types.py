from typing import Any, Callable, Deque, Dict, NamedTuple, Tuple

import jax.numpy as jnp
import numpy as onp
from jax.experimental.optimizers import OptimizerState

Key = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any
Init = Callable[[Key, Shape], Tuple[Shape, Params]]
Apply = Callable[[Params, jnp.ndarray, Dict], jnp.ndarray]
InitState = Callable[[], jnp.ndarray]
HParams = NamedTuple


class Module(NamedTuple):
    init: Init
    apply: Apply


class SchedulerState(NamedTuple):
    value: float
    metric_history: Deque[float]


class Scheduler(NamedTuple):
    step: Callable[[float, SchedulerState], SchedulerState]
    state: SchedulerState


class Optimiser(NamedTuple):
    init: Callable[[Params], OptimizerState]
    update: Callable[[int, jnp.ndarray, OptimizerState], OptimizerState]
    params: Callable[[OptimizerState], Params]


class Transition(NamedTuple):
    """
    Fields:
        x_0 (numpy.ndarray): Observation at t=0
        a_0 (numpy.ndarray): Action at t=0
        r_1 (numpy.ndarray): Reward at t=1
        x_1 (numpy.ndarray): Observation at t=1
        a_1 (numpy.ndarray) [Optional]: Action at t=0. This is an optional on-policy action
        gamma (onp.ndarray) [Optional]: Discount factor
        trace_decay (onp.ndarray) [Optional]: Eligibility trace decay for lamba returns
    """

    x_0: onp.ndarray  #  observation at t=0
    a_0: onp.ndarray  #  action at t=0
    r_1: onp.ndarray  #  reward at t=1
    x_1: onp.ndarray  #  observation at t=1
    a_1: onp.ndarray = None  #  optional on-policy action at t=1
    gamma: onp.ndarray = 1.0  #  discount factor
    trace_decay: onp.ndarray = 1.0  # eligibility trace decay for lamba returns
