import functools
from typing import Any, Callable, Dict, NamedTuple, Tuple, TypeVar, Union

import jax.numpy as jnp

Key = TypeVar("Key", bound=jnp.ndarray)
Shape = TypeVar("Shape", bound=Tuple[int, ...])
Size = TypeVar("Size", bound=Tuple[int, int])
Params = TypeVar("Params", bound=Any)
# Init = TypeVar("Init", Callable[[Key, Shape], Tuple[Shape, Params]])
# Error: TypeVar bound type cannot be generic. TODO(ep): add when generics are supported
Init = Callable[[Key, Shape], Tuple[Shape, Params]]
# Apply = TypeVar("Apply", Callable[[Params, jnp.ndarray, Dict], jnp.ndarray])
# Error: TypeVar bound type cannot be generic.
Apply = Callable[[Params, jnp.ndarray, Dict], jnp.ndarray]
HParams = TypeVar("HParams", bound=NamedTuple)
Batch = Union

State = TypeVar("State", bound=jnp.ndarray)
Observation = TypeVar("Observation", bound=jnp.ndarray)
Action = TypeVar("Action", bound=int)
Reward = TypeVar("Reward", bound=float)
Discount = TypeVar("Discount", bound=float)
TraceDecay = TypeVar("TraceDecay", bound=float)

Logits = TypeVar("Logits", bound=jnp.ndarray)
Value = TypeVar("Value", bound=jnp.ndarray)
Loss = TypeVar("Loss", bound=float)
Return = TypeVar("Return", bound=float)


def factory(cls_maker, T):
    """Type factory decorator"""

    @functools.wraps(cls_maker)
    def fabricate(*args, **kwargs):
        return T(*cls_maker(*args, **kwargs))

    return fabricate


def default(type):
    return type()


def nameof(type):
    return type.__name__
