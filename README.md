# HELX: The RL experiments framework

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![CI](https://github.com/epignatelli/helx/actions/workflows/CI.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/CI.yml)
[![CD](https://github.com/epignatelli/helx/actions/workflows/CD.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/CD.yml)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/epignatelli/helx?color=%23216477&label=Release)

**[Quickstart](#what-is-helx)** | **[Installation](#installation)** | **[Examples](#examples)** | **[Cite](#cite)**

## What is HELX?

HELX is a JAX-based ecosystem that provides a standardised framework to run Reinforcement Learning experiments.
With HELX you easily can:
- Use the `helx.envs` namespace to use the most common RL environments (gym, gymnax, dm_env, atari, ...)
- Use the `helx.agents` namespace to use the most common RL agents (DQN, PPO, SAC, ...)
- Use the `helx.experiment` namespace to run experiments on your local machine, on a cluster, or on the cloud
- Use the `helx.base` namespace to access the most common RL data structures and functions (e.g., a Ring buffer)

Each namespace provides a single, standardised interface to all agents, environments and experiment runners.

## Installation

- ### Stable
Install the stable version of `helx` and its dependencies with:
```bash
pip install helx
```

- ### Nightly
Or, if you prefer to install the latest version from source:
```bash
pip install git+https://github.com/epignatelli/helx
```



## Examples

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



## Joining development

### Adding a new agent (`helx.agents.Agent`)

An `helx` agent interface is designed as the minimal set of functions necessary to *(i)* interact with an environment and *(ii)* reinforcement learn.

```python
from typing import Any
from jax import Array

from helx.base import Timestep
from helx.agents import Agent


class NewAgent(helx.agents.Agent):
    """A new RL agent."""
    def create(self, hparams: Any) -> None:
        """Initialises the agent's internal state (knowledge), such as a table,
        or some function parameters, e.g., the parameters of a neural network."""
        # implement me

    def init(self, key: KeyArray, timestep: Timestep) -> None:
        """Initialises the agent's internal state (knowledge), such as a table,
        or some function parameters, e.g., the parameters of a neural network."""
        # implement me

    def sample_action(
        self, agent_state: AgentState, obs: Array, *, key: KeyArray, eval: bool = False
    ):
        """Applies the agent's policy to the current timestep to sample an action."""
        # implement me

    def update(self, timestep: Timestep) -> Any:
        """Updates the agent's internal state (knowledge), such as a table,
        or some function parameters, e.g., the parameters of a neural network."""
        # implement me
```


## Adding a new environment library (`helx.environment.Environment`)

To add a new library requires three steps:
1. Implement the `helx.environment.Environment` interface for the new library.
See the [dm_env](helx/environment/dm_env.py) implementation for an example.
1. Implement serialisation (to `helx`) of the following objects:
    - `helx.environment.Timestep`
    - `helx.spaces.Discrete`
    - `helx.spaces.Continuous`
2. Add the new library to the [`helx.environment.to_helx`](helx/environment/interop.py#L16) function to tell `helx` about the new protocol.

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


## A note on maintainance
This repository was born as the recipient of personal research code that was developed over the years.
Its maintainance is limited by the time and the resources of a research project resourced with a single person.
Even if I would like to automate many actions, I do not have the time to maintain the whole body of automation that a well maintained package deserves.
This is the reason of the WIP badge, which I do not plan to remove soon.
Maintainance will prioritise the code functionality over documentation and automation.

Any help is very welcome.
A quick guide to interacting with this repository:
- If you find a bug, please open an issue, and I will fix it as soon as I can.
- If you want to request a new feature, please open an issue, and I will consider it as soon as I can.
- If you want to contribute yourself, please open an issue first, let's discuss objective, plan a proposal, and open a pull request to act on it.

If you would like to be involved further in the development of this repository, please contact me directly at: `edu dot pignatelli at gmail dot com`.
