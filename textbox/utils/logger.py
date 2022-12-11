# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

"""
textbox.utils.logger
###############################
"""

import logging
import os
from accelerate.logging import get_logger
from textbox.utils.utils import ensure_dir
from collections import defaultdict
from colorama import init, Fore, Style

init(autoreset=True)

from typing import Optional


class ColorFormatter(logging.Formatter):

    FILE_FMT = "%(asctime)-15s %(levelname)s %(message)s"
    FILE_DATE_FMT = "%a %d %b %Y %H:%M:%S"
    STREAM_FMT = "%(asctime)-15s %(levelname)s %(message)s"
    STREAM_DATE_FMT = "%d %b %H:%M"

    def __init__(self, formatter_type, **kwargs):
        super().__init__(**kwargs)
        if formatter_type == 'file':
            self._formatters = defaultdict(lambda: logging.Formatter(self.FILE_FMT, self.FILE_DATE_FMT))
        else:
            self._formatters = defaultdict(
                lambda: logging.Formatter(self.STREAM_FMT, self.STREAM_DATE_FMT), {
                    logging.WARNING:
                    logging.Formatter(Fore.YELLOW + self.STREAM_FMT + Fore.RESET, self.STREAM_DATE_FMT),
                    logging.ERROR:
                    logging.Formatter(Fore.RED + self.STREAM_FMT + Fore.RESET, self.STREAM_DATE_FMT),
                    logging.CRITICAL:
                    logging.Formatter(Fore.RED + Style.BRIGHT + self.STREAM_FMT + Fore.RESET, self.STREAM_DATE_FMT),
                }
            )

    def format(self, record):
        return self._formatters[record.levelno].format(record)


def init_logger(filename: str, log_level: Optional[str], enabled: bool = True, saved_dir='saved/'):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        filename: The filename of current experiment.
        log_level: Log log_level of loggers in `logging` module.
        enabled: (Default = True) False to throttle logging output down.
        saved_dir: (Default = './log/') Directory of log files.

    Example:
        >>> init_logger("filename", "warning", disabled=True)
        >>> logger = logging.getLogger(__name__)
        >>> logger.debug("train_state")
        >>> logger.info("train_result")
        >>> logger.warning("Warning!")
        Warning!
    """
    saved_dir_name = os.path.dirname(saved_dir)
    ensure_dir(saved_dir_name)
    saved_train_dir = os.path.join(saved_dir_name, filename)
    ensure_dir(saved_train_dir)
    log_filename = 'project.log'
    log_filepath = os.path.join(saved_train_dir, log_filename)

    if log_level is None:
        log_level = "warning"
    log_level = getattr(logging, log_level.upper())

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(ColorFormatter('file'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(ColorFormatter('stream'))

    logging.basicConfig(level=log_level, handlers=[file_handler, stream_handler])
    textbox_logger = get_logger('textbox')
    textbox_logger.setLevel(log_level)
