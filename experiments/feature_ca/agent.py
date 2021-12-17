from logging import Logger
from helx.logging import NullLogger
from helx.rl.agent import IAgent
from helx.rl.policy import EGreedy
import jax
import jax.numpy as jnp
from dm_env import Environment, TimeStep


class SarsaLambda:
    """True online Sarsa(λ) algorithm for control, implemented using
    accumulating traces and linear function approximation

    Args:
    lamda (float): trace decay hyperparameter
    n_features (int): number of components to represent a state. Do not include actions.
    """

    def __init__(
        self,
        env: Environment,
        alpha: float,
        lamda: float,
        n_features: int,
        logger: Logger = NullLogger(),
    ):
        super().__init__()
        self.alpha = alpha
        self.lamda = lamda
        self.w = jnp.zeros(n_features + 1)
        self.z = jnp.zeros(n_features + 1)
        self.logger = logger
        self.policy = EGreedy(env.action_spec, 0.1)

        # temporary value estimate for the starting state
        self._actions = jnp.arange(env.action_spec().num_actions)
        self._q_old = jnp.zeros(env.action_spec().num_values)

    def phi(self, x):
        return jnp.dot(x, self.w)

    def q_values(self, s: jnp.ndarray):
        x = jnp.vmap(
            lambda a: jnp.concatenate([s, jnp.ones(s.shape[:2]) * a], axis=-1)
        )(self._actions)
        return jnp.dot(self.w.T, x)

    def select_action(self, timestep: TimeStep):
        q = self.q_values(timestep.observation)
        return self.policy.sample(q)

    def update(self, timestep: TimeStep, action: int, new_timestep: TimeStep) -> float:
        """True online Sarsa(λ) update using accumulating traces.
        See Sutton R., Barto, G. 2018. Reinforcement Learning: an Introduction, pp. 300-306"""
        g, l, a = timestep.discount, self.lamda, self.alpha
        x_m1, x = timestep.observation, new_timestep.observation
        q_m1, q = self.q_values(x_m1), self.q_values(x)

        # as we are using TD(λ) for control
        td = new_timestep.reward + g * q - q_m1

        # update traces using accumulation
        self.z = g * l * self.z + (1 - a * g * l * jnp.dot(self.z.T, x_m1)) * x_m1

        # update weights
        self.w = (
            self.w
            + a(td + q_m1 - self._q_old) * self.z
            + a * (q_m1 - self._q_old) * x_m1
        )

        # update value estimate
        error = q - self._q_old
        self._q_old = int(not new_timestep.last()) * q

        # log
        self.logger.log({"loss": error})

        return error

    def log(self, timestep, action, new_timestep, loss):
        return
