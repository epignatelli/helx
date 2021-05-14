from typing import NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
import wandb
from dm_env import specs
from jax.experimental import stax
from jax.experimental.optimizers import OptimizerState, rmsprop_momentum

from ...jax import pure
from ...nn.module import Module, module
from ...optimise.optimisers import Optimiser
from ...typing import Loss, Params, Size
from .. import td, pg
from ..memory import Trajectory
from ..agent import IAgent


class HParams(NamedTuple):
    seed: int = 0
    discount: float = 1.0
    trace_decay: float = 1.0
    n_steps: int = 1
    beta: float = 0.001
    epsilon: float = 0.2
    learning_rate: float = 0.001
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01


@module
def Cnn(n_actions: int, hidden_size: int = 512) -> Module:
    return stax.serial(
        stax.Conv(32, (8, 8), (4, 4), "VALID"),
        stax.Relu,
        stax.Conv(64, (4, 4), (2, 2), "VALID"),
        stax.Relu,
        stax.Conv(64, (3, 3), (1, 1), "VALID"),
        stax.Relu,
        stax.Flatten,
        stax.Dense(hidden_size),
        stax.Relu,
        stax.FanOut(2),
        stax.parallel(
            stax.serial(
                stax.Dense(n_actions),
                stax.Softmax,
            ),  #  actor
            stax.serial(
                stax.Dense(1),
            ),  # critic
        ),
    )


class Ppo(IAgent):
    """Proximal Policy Optiomisation algorithm.
    See:
    Schulman, J., 2017, https://arxiv.org/abs/1707.06347.
    The implementation is
    """

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        hparams: HParams,
        preprocess: lambda x: x,
    ):
        # public:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.rng = jax.random.PRNGKey(hparams.seed)
        self.preprocess = preprocess
        network = Cnn(action_spec.num_values)
        optimiser = Optimiser(
            *rmsprop_momentum(
                step_size=hparams.learning_rate,
                gamma=hparams.squared_gradient_momentum,
                momentum=hparams.gradient_momentum,
                eps=hparams.min_squared_gradient,
            )
        )
        super().__init__(network, optimiser, hparams)

        # private:
        _, params = self.network.init(self.rng, (-1, *obs_spec.shape))
        self._opt_state = self.optimiser.init(params)
        self.policy_old = params

    @pure
    def loss(
        params: Params,
        trajectory: Trajectory,
        policy_old: Params,
    ) -> Loss:
        logits, values = Ppo.network.apply(params, trajectory.x_0)
        logits_old, _ = Ppo.network.apply(policy_old, trajectory.observations)
        #  Critic loss (Bellman regression)
        gae = td.lambda_returns(trajectory, values)
        critic_loss = jnp.mean(jnp.square(gae))  #  bellman mse
        #  PPO clipped objective
        actor_loss = pg.ppo_softmax_loss(logits, logits_old, gae, Ppo.hparams)
        entropy = pg.softmax_entropy(logits)
        return actor_loss + Ppo.hparams.c1 * critic_loss - Ppo.hparams.c2 * entropy

    @pure
    def sgd_step(
        iteration: int,
        opt_state: OptimizerState,
        trajectory: Trajectory,
        policy_old: Params,
    ) -> Tuple[Loss, OptimizerState]:
        params = Ppo.optimiser.params(opt_state)
        backward = jax.value_and_grad(Ppo.loss, argnums=0)
        error, grads = backward(Ppo.network, params, trajectory, policy_old)
        return error, Ppo.optimiser.update(iteration, grads, opt_state)

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Selects an action using a softmax policy"""
        params = self.optimiser.params(self._opt_state)
        logits, _ = self.network.apply(params, timestep.observation)
        action = jax.random.categorical(self.rng, logits).squeeze()  # on-policy action
        return int(action)

    def update(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> None:
        self.buffer.add(timestep, action, new_timestep, self.preprocess)
        trajectory = self.buffer.sample(self.hparams.batch_size)
        loss, opt_state = self.sgd_step(self.network, self.optimiser, trajectory)
        return loss, opt_state

    def log(self, reward: float, loss: Loss):
        wandb.log({"Iteration": float(self._iteration)})
        wandb.log({"Reward": reward})
        if loss is not None:
            wandb.log({"Loss": float(loss)})
        return
