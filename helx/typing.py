import functools
from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax.numpy as jnp


Key = jnp.ndarray
Shape = Tuple[int, ...]
Size = Tuple[int, int]
Params = Any
Init = Callable[[Key, Shape], Tuple[Shape, Params]]
Apply = Callable[[Params, jnp.ndarray, Dict], jnp.ndarray]
HParams = NamedTuple
Logits = jnp.ndarray
Value = jnp.ndarray
Loss = float
Return = float


def factory(cls_maker, T):
    @functools.wraps(cls_maker)
    def fabricate(*args, **kwargs):
        return T(*cls_maker(*args, **kwargs))

    return fabricate
