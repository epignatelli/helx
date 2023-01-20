"""Base classes for a Reinforcement Learning agent."""
from __future__ import annotations

import abc
from pydoc import locate
from typing import Any, Callable, Generic, NamedTuple, Tuple, TypeVar
import inspect

import flax.linen as nn
import jax
from chex import Array, Shape
from jax.random import KeyArray
from optax import GradientTransformation, OptState

from ..networks.modules import Identity

from ..mdp import Action, Episode, Transition
from ..networks import AgentNetwork, apply_updates
from ..spaces import Space
from .. import __version__
from ..logging import get_logger


logging = get_logger()


class Hparams(NamedTuple):
    """A base dataclass to define the hyperparameters of an agent."""

    obs_space: Space
    action_space: Space

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


T = TypeVar("T", bound=Hparams)


class Agent(abc.ABC, Generic[T]):
    """Base class for a Reinforcement Learning agent. This class is meant to be subclassed
    for specific RL algorithms. It provides a common interface for training and evaluation
    of RL agents, and is compatible with the `helx` library for training and evaluation.
    Args:
        network (nn.Module): A Flax module that defines the network architecture.
        optimiser (GradientTransformation): An optax optimiser.
        hparams (Hparams): A dataclass that defines the hyperparameters of the agent.
        seed (int): A random seed for reproducibility.
    Properties:
        key (KeyArray): A JAX PRNG key.
        network (nn.Module): A Flax module that defines the network architecture.
        optimiser (GradientTransformation): An optax optimiser.
        hparams (Hparams): A dataclass that defines the hyperparameters of the agent.
        iteration (int): The current iteration of the agent.
        params (nn.FrozenDict): The current parameters of the agent.
        opt_state (OptState): The current optimiser state of the agent.
    """

    def __init__(
        self,
        hparams: T,
        network: AgentNetwork,
        optimiser: GradientTransformation,
        seed: int = 0,
    ):
        key: KeyArray = jax.random.PRNGKey(seed)
        obs = hparams.obs_space.sample(key)
        action = hparams.action_space.sample(key)
        outputs, params = network.init_with_output(key, obs, action, key)

        # static:
        self.seed = seed
        self.hparams: T = hparams
        self.network: AgentNetwork = network
        self.optimiser: GradientTransformation = optimiser
        self.output_shape: Shape = jax.tree_map(lambda x: x.shape, list(outputs))

        # dynamic:
        self.state = AgentState(
            params=params,
            opt_state=optimiser.init(params),
            key=key,
            step=0,
        )

    @abc.abstractmethod
    def sample_action(self, observation: Array, eval: bool = False, **kwargs) -> Action:
        """Samples an action from the agent's policy.
        Args:
            observation (Array): The observation to condition onto.
        Returns:
            Array: the action to take in the state s
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def loss(
        self, params: nn.FrozenDict, transition: Transition, **kwargs
    ) -> Tuple[Array, Any]:
        """Returns the loss of the agent on the given episode."""
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, episode: Episode) -> Array:
        """Updates the agent state at the end of each episode and returns the loss as a scalar.
        This function can used to update both the agent's parameters and its memory.
        This function is usually not jittable, as we can not ensure that the agent memory, and other
        properties are jittable. This is also a good place to perform logging."""
        raise NotImplementedError()

    def serialise(self):
        # get static fields
        ctor_args = inspect.signature(self.__init__).parameters
        static_args = [ctor_args[k].name for k in ctor_args if k != "self"]

        # get values of those fields
        init_args = {k: getattr(self, k) for k in static_args}

        # get dynamic fields
        dynamic_args = [
            k for k in self.__dict__ if k not in static_args and not k.startswith("_")
        ]

        obj = {
            "__version__": __version__,
            "__class__": self.__class__,
            "init_args": init_args,
            "dynamic_args": {k: getattr(self, k) for k in dynamic_args},
        }
        return obj

    @classmethod
    def deserialise(cls, obj):
        # read static fields
        init_args = obj["init_args"]

        # make sure we are loading the same class
        if locate(obj["__class__"]) != cls:
            logging.warning(
                "Trying to load a different class than the one that was saved, expected {}, but received {}".format(
                    cls.__name__, obj["__class__"]
                )
            )
            cls = locate(obj["__class__"])

        # make sure we are loading the same version
        if obj["__version__"] == __version__:
            logging.warning(
                "Trying to load a different version of the library, expected {}, but received {}".format(
                    __version__, obj["__version__"]
                )
            )

        # try create the object
        instance = cls.__init__(**init_args)

        # set dynamic state
        for k, v in obj["dynamic_args"].items():
            setattr(instance, k, v)

        return instance
