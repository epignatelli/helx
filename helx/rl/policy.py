from abc import ABC, abstractmethod
from dm_env import TimeStep
from dm_env.specs import BoundedArray
import jax
import jax.numpy as jnp

from helx.random import PRNGSequence


class Policy(ABC):
    @abstractmethod
    def sample(self, probs: jnp.ndarray, **kwargs):
        return

    def __call__(self, probs: jnp.ndarray, **kwargs):
        return self.sample(probs, **kwargs)


class EGreedy(Policy):
    def __init__(
        self,
        action_spec: BoundedArray,
        eps: float = 0.1,
        start_eps: float = None,
        end_eps: float = None,
        start_iter: int = None,
        end_iter: int = None,
        seed: int = 0,
    ):
        self.action_spec = action_spec
        self.eps = eps
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.rng = PRNGSequence(seed)

    def epsilon(self, iteration: int = None):
        if iteration is None:
            return self._eps

        x0, y0 = self.start_iter, self.start_eps
        x1, y1 = self.end_iter, self.end_eps
        msg = "If `iteration` is provided, the properties `start_iter`, `end_iter`, `start_eps` and `end_eps` must be set"
        assert None not in (x0, x1, y0, y1), msg

        y = ((y1 - y0) * (iteration - x0) / (x1 - x0)) + y0
        return min(max(y0, y), y1)

    def rand_prob(self):
        return jax.random.uniform(next(self.rng))

    def rand_action(self):
        return jax.random.randint(
            next(self.rng), (1,), self.action_spec.minimum, self.action_spec.maximum
        )

    def sample(self, q, iteration):
        """Draws a random action with ɛ probability,
        a greedy action with respect to q otherwise"""
        if self.rand_prob() < self.epsilon(iteration):
            return self.rand_action()
        else:
            return jnp.argmax(q, axis=-1)


class Gaussian(Policy):
    pass


class Softmax(Policy):
    pass