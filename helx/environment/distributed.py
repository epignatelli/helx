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


import logging
import multiprocessing as mp
from copy import deepcopy
from multiprocessing.connection import Connection
from typing import Any, List, Sequence

from helx.random import PRNGSequence

from ..mdp import Action, Timestep
from ..spaces import Space
from .base import Environment


def _actor(server: Connection, client: Connection, env: Environment):
    """Actor definition for Actor-Learner architectures.

    Args:
        server (Connection): server connection
        client (Connection): client connection
        env (Environment): environment to interact with
    """

    def _step(env, a: int):
        timestep = env.step(a)
        if timestep.last():
            timestep = env.reset()
        return timestep

    def _step_async(env, a: int, buffer: mp.Queue):
        timestep = env.step(a)
        buffer.put(timestep)
        print(buffer.qsize())
        if timestep.last():
            timestep = env.reset()
        return

    #  close copy of server connection from client process
    #  see: https://stackoverflow.com/q/8594909/6655465
    server.close()
    #  switch case command
    try:
        while True:
            cmd, data = client.recv()
            if cmd == "step":
                client.send(_step(env, data))
            elif cmd == "step_async":
                client.send(_step_async(env, *data))
            elif cmd == "reset":
                client.send(env.reset())
            elif cmd == "render":
                client.send(env.render())
            elif cmd == "close":
                client.send(env.close())
                break
            else:
                raise NotImplementedError("Command {} is not implemented".format(cmd))
    except KeyboardInterrupt:
        logging.info("SubprocVecEnv actor: got KeyboardInterrupt")
    finally:
        env.close()


class MultiprocessEnv(Environment):
    """
    This class is inspired by openai's SubprocEnv.
    https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
    An environment that allows concurrent interactions to improve experience collection throughput, as used in:
    https://arxiv.org/abs/1602.01783, https://arxiv.org/abs/1802.01561 and https://arxiv.org/abs/1707.06347
    The class runs multiple subproceses and communicates with them via pipes.
    """

    def __init__(
        self,
        env: Environment,
        n_actors: int,
        context: str = "spawn",
        seed: int = 0,
    ):
        assert isinstance(
            env, Environment
        ), "The environment to parallelise must be an instance of `helx.Environment`, got {} instead".format(
            type(env)
        )
        #  public:
        self.n_actors: int = n_actors
        self.clients: Sequence[Connection] = []
        self.servers: Sequence[Connection] = []
        self.envs: Sequence[Environment] = [deepcopy(env) for _ in range(n_actors)]
        self.processes = []

        #  setup parallel workers
        rng = PRNGSequence(seed)
        ctx = mp.get_context(context)
        pipes = zip(*[ctx.Pipe() for _ in range(self.n_actors)])
        self.clients, self.servers = pipes
        for server, client, env in zip(self.servers, self.clients, self.envs):
            env.seed(int(next(rng)[0]))
            self.processes.append(
                ctx.Process(  # type: ignore
                    target=_actor,
                    args=(server, client, env),
                    daemon=True,
                )
            )

        for i, p in enumerate(self.processes):
            logging.info("Starting actor {} on process {}".format(i, p))
            p.start()
            #  close copy of client connection from server process
            #  see: https://stackoverflow.com/q/8594909/6655465
            self.clients[i].close()

    def __del__(self):
        self.close()

    def action_space(self) -> Space:
        return self.envs[0].action_space()

    def observation_space(self) -> Space:
        return self.envs[0].observation_space()

    def reward_space(self) -> Space:
        return self.envs[0].reward_space()

    def reset(self) -> List[Any]:
        for server in self.servers:
            server.send(("reset", None))
        return self._receive()

    def step(self, actions: Sequence[Action]) -> List[Timestep]:
        self._check_actions(actions)
        for a, server in zip(actions, self.servers):
            server.send(("step", a))
        return self._receive()

    def step_async(self, actions: Sequence[Action], queue: mp.Queue) -> None:
        self._check_actions(actions)
        for a, server in zip(actions, self.servers):
            server.send(("step_async", (a, queue)))
        return

    def close(self) -> List[Any]:
        for server in self.servers:
            server.send(("close", None))
        for p in self.processes:
            p.join()
        return self._receive()

    def render(self, mode: str = "rgb_array"):
        for server in self.servers:
            server.send(("render", mode))
        return self._receive()

    def is_waiting(self):
        return any(server.poll() for server in self.servers)

    def _receive(self) -> List[Any]:
        return [server.recv() for server in self.servers]

    def _check_actions(self, actions: Sequence[Action]):
        assert (
            len(actions) == self.n_actors
        ), "The number of actions must be equal to the number of parallel environments.\
            \nReceived {} actions for {} environments. ".format(
            len(actions), self.n_actors
        )
