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

from __future__ import annotations

import logging
import abc
from typing import Any, Dict, List
import os
from chex import Array

import pandas as pd
import wandb
import tensorboardX


BLACK = "\033[0;30m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
BROWN = "\033[0;33m"
BLUE = "\033[0;34m"
PURPLE = "\033[0;35m"
CYAN = "\033[0;36m"
GREY = "\033[0;37m"

DARK_GREY = "\033[1;30m"
LIGHT_RED = "\033[1;31m"
LIGHT_GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
LIGHT_BLUE = "\033[1;34m"
LIGHT_PURPLE = "\033[1;35m"
LIGHT_CYAN = "\033[1;36m"
WHITE = "\033[1;37m"
BOLD_RED = "\033[31;1m"
BOLD = "\033[1m"
RESET = "\033[0m"

_logger = None


def get_default_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger("helx")
        _logger.propagate = False
        _logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(Formatter())
        _logger.addHandler(ch)
    return _logger


class Formatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: DARK_GREY,
        logging.INFO: GREY,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record):
        level_fmt = self.FORMATS.get(record.levelno)
        format = (f"{WHITE}%(name)s{RESET}"
                f"{level_fmt}[%(levelname)s]{RESET} "
                f"{WHITE}%(asctime)s{RESET} "
                f"{WHITE}%(filename)s:%(lineno)d:{RESET} "
                f"{level_fmt}%(message)s{RESET}")
        formatter = logging.Formatter(format, datefmt="%H:%M:%S")
        return formatter.format(record)


class Logger(abc.ABC):
    def __init__(
        self,
        experiment_name: str,
        log_frequency: int,
        terminal_logger: logging.Logger = get_default_logger(),
    ):
        self.experiment_name = experiment_name
        self.log_frequency = log_frequency
        self.debug = False
        self._terminal_logger = terminal_logger

    @abc.abstractmethod
    def record(self, record: Dict[str, Any]):
        raise NotImplementedError()

    def record_image(self, name: str, image: Array, step: int):
        raise NotImplementedError()

    def record_video(self, name: str, video: Array, step: int):
        raise NotImplementedError()

    def record_audio(self, name: str, audio: Array, step: int):
        raise NotImplementedError()

    def record_histogram(self, name: str, histogram: Array, step: int):
        raise NotImplementedError()

    def record_graph(self, name: str, graph: Any, step: int):
        raise NotImplementedError()

    def log(self, message: str, level: int = logging.INFO):
        self._terminal_logger.log(level, message)

    def debug_mode(self):
        self.debug = True
        return

    def non_debug_mode(self):
        self.debug = False
        return

    def __add__(self, other: Logger):
        return CompoundLogger([self, other])


class CompoundLogger(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def record(self, record: Dict[str, Any]):
        for logger in self.loggers:
            logger.record(record)

    def debug_mode(self):
        for logger in self.loggers:
            logger.debug_mode()


class NullLogger(Logger):
    def __init__(self, experiment_name: str="default", log_frequency: int = 1):
        super().__init__(experiment_name, log_frequency)

    def record(self, record: Dict[str, Any]):
        return


class CsvLogger(Logger):
    def __init__(self, experiment_name: str, log_frequency: int = 1, folder: str = ""):
        super().__init__(experiment_name, log_frequency)
        path = os.path.join(folder, f"{experiment_name}.csv")
        if os.path.exists(path):
            raise FileExistsError(f"File {path} already exists, plaese specify another name or path.")

        self.path = path
        self.records = list()

    def record(self, record: Dict[str, Any]):
        self.records.append(record)
        pd.DataFrame(self.records).to_csv(self.path, index=False)


class JsonLogger(Logger):
    def __init__(self, experiment_name: str, log_frequency: int = 1, folder: str = ""):
        super().__init__(experiment_name, log_frequency)
        path = os.path.join(folder, f"{experiment_name}.json")
        if os.path.exists(path):
            raise FileExistsError(f"File {path} already exists, plaese specify another name or path.")

        self.path = path
        self.records = list()

    def record(self, record: Dict[str, Any]):
        self.records.append(record)
        pd.DataFrame(self.records).to_json(self.path, index=False)


class WAndBLogger(Logger):
    def __init__(
        self,
        experiment_name: str = "debug",
        log_frequency: int = 1,
        project="helx",
        mode: str = "online",
        **kwargs,
    ):
        super().__init__(experiment_name, log_frequency)
        wandb.init(
            project=project,
            name=self.experiment_name,
            reinit=True,
            mode=mode,
            **kwargs,
        )

    def record(self, record: Dict[str, Any]):
        return wandb.log(record)


class TensorboardXLogger(Logger):
    def __init__(
        self,
        experiment_name: str = "debug",
        log_frequency: int = 1,
        folder: str = "",
        **kwargs,
    ):
        super().__init__(experiment_name, log_frequency)
        self.log_dir = os.path.join(folder, f"{experiment_name}")
        self.writer = tensorboardX.SummaryWriter(self.log_dir, **kwargs)
        self.step = 0

    def record(self, record: Dict[str, Any]):
        for key, value in record.items():
            self.writer.add_scalar(key, value, self.step)
        self.step += 1
