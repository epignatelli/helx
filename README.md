# HELX: The RL experiments framework

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![CI](https://github.com/epignatelli/helx/actions/workflows/CI.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/CI.yml)
[![CD](https://github.com/epignatelli/helx/actions/workflows/CD.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/CD.yml)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/epignatelli/helx?color=%23216477&label=Release)

**[Quickstart](#what-is-helx)** | **[Installation](#installation)** | **[Examples](#examples)** | **[Cite](#cite)**

## What is HELX?

HELX is a JAX-based ecosystem that provides a standardised and ready to use framework to run Reinforcement Learning experiments.

The ecosystem is made of three main nodes:
- [Helx](https://github.com/epignatelli/helx), containing the base objects of the framework
- [Agentx](https://github.com/epignatelli/agentx), containing the most common RL baselines
- [Envx](https://github.com/epignatelli/envx), which provides a single API to the most common RL environments (gym, gymnax, dm_env, atari, ...)
- [Expx](https://github.com/epignatelli/expx), which packs the other three nodes into a framework to run and manage experiments

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


---
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

---
## Supported libraries

We currently support these external environment models:
- [dm_env](https://github.com/deepmind/dm_env)
- [bsuite](https://github.com/deepmind/bsuite)
- [dm_control](https://github.com/deepmind/dm_control), including
  - [Mujoco](https://mujoco.org)
- [gym](https://github.com/openai/gym) and [gymnasium](https://github.com/Farama-Foundation/Gymnasium), including
  - The [minigrid]() family
  - The [minihack]() family
  - The [atari](https://github.com/mgbellemare/Arcade-Learning-Environment) family
  - The legacy [mujoco](https://www.roboti.us/download.html) family
  - And the standard gym family
- [gym3](https://github.com/openai/gym3), including
  - [procgen](https://github.com/openai/procgen)

#### On the road:
- [gymnax](https://github.com/RobertTLange/gymnax)
- [ivy_gym](https://github.com/unifyai/gym)
---
## Adding a new agent (`helx.agents.Agent`)

An `helx` agent interface is designed as the minimal set of functions necessary to *(i)* interact with an environment and *(ii)* reinforcement learn.

```python
class Agent(ABC):
    """A minimal RL agent interface."""

    @abstractmethod
    def sample_action(self, timestep: Timestep) -> Array:
        """Applies the agent's policy to the current timestep to sample an action."""

    @abstractmethod
    def update(self, timestep: Timestep) -> Any:
        """Updates the agent's internal state (knowledge), such as a table,
        or some function parameters, e.g., the parameters of a neural network."""
```

---
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

---
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
