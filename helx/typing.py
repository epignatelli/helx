import functools
from typing import Any, Callable, Dict, NamedTuple, Tuple, TypeVar

import jax.numpy as jnp

Key = TypeVar("Key", jnp.ndarray)
Shape = TypeVar("Shape", Tuple[int, ...])
Size = TypeVar("Size", Tuple[int, int])
Params = TypeVar("Params", Any)
# Init = TypeVar("Init", Callable[[Key, Shape], Tuple[Shape, Params]])
# Error: TypeVar bound type cannot be generic. TODO(ep): add when generics are supported
Init = Callable[[Key, Shape], Tuple[Shape, Params]]
# Apply = TypeVar("Apply", Callable[[Params, jnp.ndarray, Dict], jnp.ndarray])
# Error: TypeVar bound type cannot be generic.
Apply = Callable[[Params, jnp.ndarray, Dict], jnp.ndarray]
HParams = TypeVar("HParams", NamedTuple)
Logits = TypeVar("Logits", jnp.ndarray)
Value = TypeVar("Value", jnp.ndarray)
Loss = TypeVar("Loss", float)
Return = TypeVar("Return", float)


def factory(cls_maker, T):
    @functools.wraps(cls_maker)
    def fabricate(*args, **kwargs):
        return T(*cls_maker(*args, **kwargs))

    return fabricate
