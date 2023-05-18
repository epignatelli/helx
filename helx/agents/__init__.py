# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the collection of learners for RL algorithms (agents).
Different implementations of learners can yield synchronous or asynchronous agents,
and can be used for single or multi-agent learning.
"""

from .agent import Agent
from .a2c import A2C, A2CHparams
from .dqn import DQN, DQNHparams
from .sac import SAC, SACHparams
from .sacd import SACD
