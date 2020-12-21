![Build](https://github.com/epignatelli/helx/workflows/build/badge.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Helx
Helx is a helper library for [JAX](https://github.com/google/jax) / [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py).
It is in continuous development, so expect changes.
For some fuctionalities see below.



### Installation
To run the algorithm on a GPU, I suggest to [install](https://github.com/google/jax#pip-installation) the gpu version of `jax` [[4]](https://github.com/google/jax). You can then install this repo using [Anaconda python](https://www.anaconda.com/products/individual) and [pip](https://pip.pypa.io/en/stable/installing/).
```sh
pip install git+https://github.com/epignatelli/helx
```
Note that helx does not depend on JAX, even if it uses it. This allows you to use a version of JAX built for an arbitrary accelerator.



### Using `module`
A `module` is a simple interface for stax functions. It returns a standardised `NamedTuple` for a layer construction function with an `init` and an `apply` function. Here's an example:
```python
import jax
from jax.experimental.stax import Dense, Relu
from helx import Module, module


@module
def Mlp() -> Module:
    return serial(
        Dense(16),
        Relu,
        Dense(32),
        Relu,
        Dense(8)
    )

rng = jax.random.PRNGKey(0)
input_shape = (128, 8)
x = jax.random.normal(input_shape)

mpl = Mpl()
output_shape, params = mlp.init(rng, input_shape)
y_hat = mpl.apply(params, x)
)


```

### Using the `inject` decorator
The `inject` decorator allows you to define and use a pure function in a class.

Reusing the mlp example above:
```python
import jax
from jax.experimental.optimizer import sdg
from helx import Module, module, inject


class Agent:
    def __init__(self, network, optimiser, input_shape, seed):
        self.iteration = 0
        self.rng = jax.random.PRNGKey(seed)
        params = network.init(rng, input_shape)
        self.optimiser_state = optimiser.init_fn(self.params)

        @inject
        def forward(
            params: Params,
            trajectory: Trajectory,
        ) -> jnp.ndarray:
            outputs = self.model.apply(trajectory.observations, prev_state)
            return outputs

        @partial(inject, static_argnums=(0, 1))
        def sgd_step(
            iteration: int,
            optimiser_state: OptimizerState,
            trajectory: jnp.ndarray,
        ) -> Tuple[float, OptimizerState]:
            params = optimiser.params_fn(optimiser_state)
            grads, outputs = jax.grad(forward)(params, trajectory)
            optimiser_state = optimiser.update_fn(iteration, grads, optimiser_state)
            return outputs, optimiser_state

    def update(self, trajectory):
        return self.sgd_step(self.iteration, self.optimiser_state, trajectory)


mpl = Mpl()
agent = Agent(mpl, adam(1e-3))

output_shape, params = mlp.init(rng, input_shape)

# you can call forward as a classmethod
agent.forward(params, x)
agent.sgd_step(params, x)

# or use a class method that calls an injected function
agent.update(x)

```