import pickle
from typing import NamedTuple

import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.optimizers import OptimizerState

from .distributed import redistribute_tree
from .types import HParams


class TrainState(NamedTuple):
    iteration: int
    opt_state: OptimizerState
    rng: jnp.ndarray = None
    hparams: HParams = None

    def serialise(self):
        state = self._replace(
            opt_state=optimizers.unpack_optimizer_state(self.opt_state)
        )
        return pickle.dumps(state)

    @staticmethod
    def deserialise(obj):
        state = pickle.loads(obj)
        opt_state = optimizers.pack_optimizer_state(state.opt_state)
        state = state._replace(opt_state=redistribute_tree(opt_state))
        return state

    def save(self, filepath):
        with open(filepath, "wb") as f:
            state = self.serialise()
            f.write(state)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return TrainState.deserialise(f.read())
