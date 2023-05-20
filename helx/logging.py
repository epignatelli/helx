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
import wandb


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
RESET = "\033[0m"
BOLD_RED = "\033[31;1m"


_logger = None


class Logger(abc.ABC):
    def __init__(self, experiment_name: str, log_frequency: int):
        self.experiment_name = experiment_name
        self.log_frequency = log_frequency
        self.debug = False

    @abc.abstractmethod
    def record(self, record: Dict[str, Any]):
        raise NotImplementedError()

    @abc.abstractmethod
    def log(self, message: str):
        raise NotImplementedError()

    def debug_mode(self):
        self.debug = True
        return

    def __add__(self, other: Logger):
        return CompoundLogger([self, other])


class CompoundLogger(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def record(self, record: Dict[str, Any]):
        for logger in self.loggers:
            logger.record(record)

    def log(self, message: str):
        for logger in self.loggers:
            logger.log(message)

    def debug_mode(self):
        for logger in self.loggers:
            logger.debug_mode()


class StreamLogger(Logger):
    def __init__(self, experiment_name: str = "debug", log_frequency: int = 10):
        super().__init__(experiment_name, log_frequency)
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        ch = ColorizingStreamHandler()
        ch.setLevel(logging.DEBUG)
        self.logger.addHandler(ch)

    def record(self, record: Dict[str, Any]):
        self.logger.info(str(record))

    def log(self, message: str):
        self.logger.info(message)


def get_default_logger():
    global _logger
    if _logger is None:
        _logger = StreamLogger("helx")
    return _logger


class WAndBLogger(Logger):
    def __init__(
        self,
        experiment_name: str = "debug",
        log_frequency: int = 1,
        project="helx",
        mode: str = "online"
    ):
        super().__init__(experiment_name, log_frequency)
        wandb.init(
            project=project,
            name=self.experiment_name,
            reinit=True,
            mode=mode,
        )

    def record(self, record: Dict[str, Any]):
        return wandb.log(record)

    def log(self, message: str):
        return wandb.log(message)


class CustomFormatter(logging.Formatter):

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"  # pyright: ignore reportGeneralTypeIssues

    FORMATS = {
        logging.DEBUG: GREY + str(format) + RESET,
        logging.INFO: GREY + str(format) + RESET,
        logging.WARNING: YELLOW + str(format) + RESET,
        logging.ERROR: RED + str(format) + RESET,
        logging.CRITICAL: BOLD_RED + str(format) + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ColoredFormatter(logging.Formatter):
    """Special custom formatter for colorizing log messages!"""

    def __init__(self, *args, **kwargs):
        self._colors = {
            logging.DEBUG: DARK_GREY,
            logging.INFO: RESET,
            logging.WARNING: BROWN,
            logging.ERROR: RED,
            logging.CRITICAL: LIGHT_RED,
        }
        super(ColoredFormatter, self).__init__(*args, **kwargs)

    def format(self, record):
        """Applies the color formats"""
        record.msg = self._colors[record.levelno] + record.msg + RESET
        return logging.Formatter.format(self, record)

    def setLevelColor(self, logging_level, escaped_ansi_code):
        self._colors[logging_level] = escaped_ansi_code


class ColorizingStreamHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        self._colors = {
            logging.DEBUG: DARK_GREY,
            logging.INFO: RESET,
            logging.WARNING: BROWN,
            logging.ERROR: RED,
            logging.CRITICAL: LIGHT_RED,
        }
        super(ColorizingStreamHandler, self).__init__(*args, **kwargs)

    @property
    def is_tty(self):
        isatty = getattr(self.stream, "isatty", None)
        return isatty and isatty()

    def emit(self, record):
        try:
            message = self.format(record)
            stream = self.stream
            if not self.is_tty:
                stream.write(message)
            else:
                message = self._colors[record.levelno] + message + RESET
                stream.write(message)
            stream.write(getattr(self, "terminator", "\n"))
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def setLevelColor(self, logging_level, escaped_ansi_code):
        self._colors[logging_level] = escaped_ansi_code
