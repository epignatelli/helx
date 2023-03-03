[![Continuous Integration](https://github.com/epignatelli/helx/actions/workflows/CI.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/CI.yml)
[![Continuous deployment](https://github.com/epignatelli/helx/actions/workflows/CD.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/CD.yml)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)

[**Quickstart**](#quickstart)
| [**Install guide**](#Installation)
| [**Contributing**](#contributing)
| [**Contacts**](#contacts)
| [**Cite**](#cite)

## What is HELX?
Helx is a backend-agnostic Deep RL library, aiming to make it easier for researchers to **implement** new algorithms, **benchmark** existing ones, and **reproduce** experiments for reliable evaluation and advancement of the Deep RL field.


## Why helx?
Algorithm implementation is a key part of a Deep RL methods.
The same method, implemented differently, yields very different results []() .
Experiments are, at times, partial, cherry-picked and hard to reproduce []().

Our goals are simple: we want to make it easier for researchers to test their new algorithms, compare them to existing ones, and evaluate their results in a reliable way.
We need Deep RL algorithms implementations that are:
1. Reliable
2. Interpretable
3. Modular
4. Back-end agnostic
5. Co-created, community-driven and agreed upon


To achieve this, we provide a stable **front-end interfaces** while remaining **agnostic** to the **backends** used by these components:
- **`Agent`**,
- **`Environment`**, and an
- **`Experiment`**

This makes it possible to integrate new algorithms easily, and to test them across a variety of environments, with minimal effort.

## Quickstart

A typical use case is to design an agent, and toy-test it on `catch` before evaluating it on more complex environments, such as atari, procgen or mujoco.

```python
import bsuite
import gym

import helx.environment
import helx.experiment
import helx.agents

# create the enviornment in you favourite way
env = bsuite.load_from_id("catch/0")
# convert it to an helx environment
env = helx.environment.to_helx(env)
# create the agent
hparams = helx.agents.Hparams(env.obs_space(), env.action_space())
agent = helx.agents.Random(hparams)

# run the experiment
helx.experiment.run(env, agent, episodes=100)
```


Switching to a different environment is as simple as changing the `env` variable.


```diff
import bsuite
import gym

import helx.environment
import helx.experiment
import helx.agents

# create the enviornment in you favourite way
-env = bsuite.load_from_id("catch/0")
+env = gym.make("procgen:procgen-coinrun-v0")
# convert it to an helx environment
env = helx.environment.to_helx(env)
# create the agent
hparams = helx.agents.Hparams(env.obs_space(), env.action_space())
agent = helx.agents.Random(hparams)

# run the experiment
helx.experiment.run(env, agent, episodes=100)
```


## Installation
To use `helx` as it is, you can install it from `pip` and `pypi` with:
```bash
pip install helx
```

If you also want to download the binaries for `mujoco`, both `gym` and `dm_control`, and `atari`:
```bash
helx-download-extras
```

And then tell the system where the mujoco binaries are:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mujoco/lib
export MJLIB_PATH=/path/to/home/.mujoco/mujoco210/bin/libmujoco210.so
export MUJOCO_PY_MUJOCO_PATH=/path/to/home/.mujoco/mujoco210
```

## Contributing
Please see our [contributing guidelines](CONTRIBUTING.md) for how **how** to contribute.
You can either contribute in maintaining the codebase, or by incrementing the list of:
- agents,
- supported environments and environment suites,
- experiments

### 1. Adding a new agent
An `helx.agents.Agent` interface is designed as the minimal set of functions necessary to
- interact with an environment and
- reinforcement learn through it

For example, a random agent can be implemented as follows:
```python
class RandomAgent(helx.agents.Agent):
    """A minimal RL agent interface."""

    def sample_action(self, timestep: Timestep) -> Action:
        """Applies the agent's policy to the current timestep to sample an action."""
        # using jax
        return jax.random.randint(self.key, (), 0, self.n_actions)
        # or pytorch
        return torch.rand(self.n_actions)
        # or even numpy
        return np.random.randint(0, self.n_actions)

    def update(self, timestep: Timestep) -> Any:
        """Updates the agent's internal state (knowledge), such as a table,
        or some function parameters, e.g., the parameters of a neural network."""
        return None
```

## 2. Adding a new environment
An `helx.environment.Environment` is a synthesis of the minimal set of functions necessary arising from different RL environment interfaces, such as `gym` and `dm_env`.

For example, a random environment can be implemented as follows:
```python
class RandomEnvironment(helx.environment.Environment):
    """A minimal RL agent interface."""

    def action_space(self) -> Space:
        return helx.spaces.Discrete(2)

    def observation_space(self) -> Space:
        return helx.spaces.Continuous((3, 16, 16), minimum=0, maximum=1)

    def reward_space(self) -> Space:
        return helx.spaces.Continuous((), minimum=0, maximum=1)

    def state(self) -> Array:
        return self._current_state

    def reset(self, seed: int | None = None) -> Timestep:
        return self.observation_space().sample()

    @abc.abstractmethod
    def step(self, action: Action) -> Timestep:
        return self.observation_space().sample()
```



---
## Cite
If you use `helx` please consider citing it as:

```bibtex
@misc{helx,
  author = {Pignatelli, Eduardo},
  title = {Helx: Interoperating between Reinforcement Learning Experimental Protocols},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/epignatelli/helx}}
  }
```

---
## A note on maintainance
This repository was born as the recipient of personal research code that was developed over the years.
Its maintainance is limited by the time and the resources of a research project resourced with a single person.
Even if I would like to automate many actions, I do not have the time to maintain the whole body of automation that a well maintained package deserves.
This is the reason of the `alpha` badge, which I do not plan to remove soon.
Maintainance will prioritise the code functionality over documentation and automation.

Any help is very welcome.
A quick guide to interacting with this repository:
- If you find a bug, please open an issue, and I will fix it as soon as I can.
- If you want to request a new feature, please open an issue, and I will consider it as soon as I can.
- If you want to contribute yourself, please open an issue first, let's discuss objective, plan a proposal, and open a pull request to act on it.

If you would like to be involved further in the development of this repository, please contact me directly at: `edu dot pignatelli at gmail dot com`.
