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


def get_logger():
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("helx")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = ColorizingStreamHandler()
    ch.setLevel(logging.DEBUG)

    # add ch to logger
    logger.addHandler(ch)

    # set logger
    _logger = logger

    return logger


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
