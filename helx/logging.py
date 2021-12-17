import logging
from abc import ABC, abstractmethod
from logging import handlers
from datetime import datetime
from typing import Sequence, Tuple

import wandb


class Logger(ABC):
    @abstractmethod
    def log(self, *args, **kwargs):
        return


class NullLogger(Logger):
    def log(self):
        return None


class TerminalLogger(Logger):
    def __init__(self, level: int = 20):
        """
        Uses the python default logging module to print
        information on the screen.

        Args:

        level (int): minimum logging level, between:
            CRITICAL = 50
            ERROR = 40
            WARNING = 30
            INFO = 20
            DEBUG = 10
            NOTSET = 0
        """
        logger = logging.getLogger("helx-terminal")
        logger.setLevel(level)
        self._logger = logger

    def log(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)


class FileLogger(Logger):
    def __init__(self, filepath: str, level: int = 20):
        """
        Uses the python default logging module to print
        information on the screen.

        Args:

        filepath (str): The path of the file to save the logging to
        level (int): minimum logging level, between:
            CRITICAL = 50
            ERROR = 40
            WARNING = 30
            INFO = 20
            DEBUG = 10
            NOTSET = 0
        """
        logger = logging.getLogger("helx-file")
        logger.setLevel(level)
        logger.addHandler(handlers.RotatingFileHandler(filepath, maxBytes=2e8))
        self._logger = logger

    @property
    def filepath(self):
        if self._logger.hasHandlers():
            return None
        return self._logger.handlers[0].baseFilename

    def log(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)


class EmailLogger(Logger):
    def __init__(
        self,
        mailhost: Tuple[str, int],
        sender: str,
        recipients: Sequence[str],
        credentials: Tuple[str, str],
        level=20,
    ):
        # TODO
        raise NotImplementedError


class HttpLogger(Logger):
    def __init__(self, host: str, method: str = "POST", level: int = 20):
        # TODO
        raise NotImplementedError


class WandBLogger(Logger):
    def __init__(self, *args, **kwargs):
        wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        return wandb.log(*args, **kwargs)


class TensorboardLogger(Logger):
    # TODO
    raise NotImplementedError
