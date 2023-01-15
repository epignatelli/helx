[![Test](https://github.com/epignatelli/helx/actions/workflows/test.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/linting.yml)
[![Examples](https://github.com/epignatelli/helx/actions/workflows/run_examples.yml/badge.svg)](https://github.com/epignatelli/helx/actions/workflows/run_examples.yml)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

--------------

# Helx

Helx provides a single interface to a) interoperate between a variety of Reinforcement Learning environment and to b) code interacting agents.
It is an interface to jumpstart experiments


## Installation
```bash
pip install git+https://github.com/epignatelli/helx
```
---
## Example

A typical use case is to design an agent, and toy-test it on `catch` before evaluating it on more complex environments, such as atari, procgen or mujoco.

```python
import bsuite
import gym

import helx.environment
import helx.experiment
from helx.agents import RandomAgent, Hparams

# create the enviornment in you favourite way
env = bsuite.load_from_id("catch/0")
# convert it to an helx environment
env = helx.environment.make_from(env)
# create the agent
hparams = Hparams(env.obs_space(), env.action_space())
agent = RandomAgent(hparams)

# run the experiment
helx.experiment.run(env, agent, episodes=100)
```


Switching to a different environment is as simple as changing the `env` variable.
```diff
import bsuite
import gym

import helx.environment
import helx.experiment
from helx.agents import RandomAgent, Hparams

# create the enviornment in you favourite way
-env = bsuite.load_from_id("catch/0")
+env = gym.make("procgen:procgen-coinrun-v0")
# convert it to an helx environment
env = helx.environment.make_from(env)
# create the agent
hparams = Hparams(env.obs_space(), env.action_space())
agent = RandomAgent(hparams)

# run the experiment
helx.experiment.run(env, agent, episodes=100)
```

---
## Supported libraries

We currently support the current external environment models:
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

---
## Adding a new environment library

Adding a new library requires three steps:
1. Implement the `helx.environment.Environment` interface for the new library.
See the [dm_env](helx/environment/dm_env.py) implementation for an example.
1. Implement (de)serialisation of the following objects:
    - `helx.environment.Timestep`
    - `helx.spaces.Discrete`
    - `helx.spaces.Continuous`
2. Add the new library to the [`helx.environment.make_from`](helx/environment/interop.py) function to tell `helx` about the new protocol.

---
## The `helx.agents.Agent` interface

An `helx` agent interface is designed as the minimal set of functions to interact with an environment and learning.

```python
class Agent(ABC):
    """An agent interface."""

    @abstractmethod
    def update(self, timestep: Timestep) -> Any:
        """Updates the agent's internal state (knowledge), such as a table,
        or some function parameters, e.g., the parameters of a neural network."""

    @abstractmethod
    def sample_action(self, timestep: Timestep) -> Action:
        """Applies the agent's policy to the current timestep to sample an action."""
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
