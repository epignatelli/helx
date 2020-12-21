from typing import Callable, Dict, NamedTuple, Tuple, Any
import jax.numpy as jnp


RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any
Init = Callable[[RNGKey, Shape], Tuple[Shape, Params]]
Apply = Callable[[Params, jnp.ndarray, Dict], jnp.ndarray]
InitState = Callable[[], jnp.ndarray]


class Module(NamedTuple):
    init: Init
    apply: Apply
    initial_state: InitState = None