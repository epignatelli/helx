# pyright: reportGeneralTypeIssues=false
from functools import singledispatch
from typing import Any, Callable, cast

from gym.utils.step_api_compatibility import (
    convert_to_terminated_truncated_step_api as gym_convert_to_terminated_truncated_step_api,
    TerminatedTruncatedStepType as GymTimestep,
)
from gymnasium.utils.step_api_compatibility import (
    convert_to_terminated_truncated_step_api as gymnasium_convert_to_terminated_truncated_step_api,
    TerminatedTruncatedStepType as GymnasiumTimestep,
)
import dm_env.specs
import jax.numpy as jnp

from .mdp import StepType, Timestep
from .spaces import Continuous, Discrete, Space


def polymorph(function: Callable[..., Any]) -> Callable[..., Any]:
    """Implements polymorphic dispatching for the to_helx function."""
    # wrap the function
    wrapped = singledispatch(function)
    # inject the method into the class
    cls = function.__globals__.get(function.__qualname__.split(".")[0])
    if cls is not None:
        setattr(cls, function.__name__, wrapped)
    return wrapped


def as_dynamic(instance: Any) -> Any:
    return cast(type(instance), instance)


class UnsupportedConversion(TypeError):
    def __init__(self, instance):
        super().__init__("Unsupported object type conversion {}".format(type(instance)))


@polymorph
def to_helx(space: dm_env.specs.BoundedArray) -> Continuous:
    return Continuous(
        space.shape, space.dtype, jnp.asarray(space.minimum), jnp.asarray(space.maximum)
    )


@polymorph
def to_helx(space: dm_env.specs.DiscreteArray) -> Discrete:
    return Discrete(space.num_values)


@polymorph
def to_helx(space: dm_env.specs.Array) -> Continuous:
    try:
        return to_helx(as_dynamic(space))
    except:
        raise UnsupportedConversion(space)
