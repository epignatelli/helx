from typing import Any, Callable, Dict, NamedTuple, Tuple, Deque
import numpy as onp
import jax.numpy as jnp
from jax.experimental.optimizers import OptimizerState


RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any
Init = Callable[[RNGKey, Shape], Tuple[Shape, Params]]
Apply = Callable[[Params, jnp.ndarray, Dict], jnp.ndarray]
InitState = Callable[[], jnp.ndarray]


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
    s: onp.ndarray
    a: onp.ndarray
    r: onp.ndarray
    ns: onp.ndarray
    na: onp.ndarray = None  #  optional on-policy action
