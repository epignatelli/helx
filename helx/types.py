from typing import Any, Callable, Deque, Dict, NamedTuple, Tuple

import jax.numpy as jnp
import numpy as onp
from jax.experimental.optimizers import OptimizerState


Key = jnp.ndarray
Shape = Tuple[int, ...]
Size = Tuple[int, int]
Params = Any
Init = Callable[[Key, Shape], Tuple[Shape, Params]]
Apply = Callable[[Params, jnp.ndarray, Dict], jnp.ndarray]
InitState = Callable[[], jnp.ndarray]
HParams = NamedTuple
Logits = jnp.ndarray
Value = jnp.ndarray
Loss = float
Return = float


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
