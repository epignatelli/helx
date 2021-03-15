from typing import Any, Callable, Dict, NamedTuple, Tuple
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


class Optimiser(NamedTuple):
    init: Callable[[Params], OptimizerState]
    update: Callable[[OptimizerState], OptimizerState]
    params: Callable[[OptimizerState], Params]


class Transition(NamedTuple):
    s: onp.ndarray
    a: onp.ndarray
    r: onp.ndarray
    ns: onp.ndarray
    na: onp.ndarray = None  #  optional on-policy action
