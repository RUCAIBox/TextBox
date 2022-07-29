# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

"""
textbox.utils.logger
###############################
"""

import logging
import os
from textbox.utils.utils import ensure_dir

from typing import Optional


FILE_FMT = "%(asctime)-15s %(levelname)s %(message)s"
FILE_DATE_FMT = "%a %d %b %Y %H:%M:%S"
STREAM_FMT = "%(asctime)-15s %(levelname)s %(message)s"
STREAM_DATE_FMT = "%d %b %H:%M"


def init_logger(filename: str, log_level: Optional[str], enabled: bool = True, logdir: str = './log/'):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        filename: The filename of current experiment.
        log_level: Log log_level of loggers in `logging` module.
        enabled: (Default = True) False to throttle logging output down.
        logdir: (Default = './log/') Directory of log files.

    Example:
        >>> init_logger("filename", "warning", disabled=True)
        >>> logger = logging.getLogger(__name__)
        >>> logger.debug("train_state")
        >>> logger.info("train_result")
        >>> logger.warning("Warning!")
        Warning!
    """
    dir_name = os.path.dirname(logdir)
    ensure_dir(dir_name)

    log_filename = filename + '.log'
    log_filepath = os.path.join(logdir, log_filename)

    file_formatter = logging.Formatter(FILE_FMT, FILE_DATE_FMT)
    stream_formatter = logging.Formatter(STREAM_FMT, STREAM_DATE_FMT)

    if log_level is None:
        log_level = "warning"
    log_level = getattr(logging, log_level.upper(), None)

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(stream_formatter)

    logging.basicConfig(level=log_level, handlers=[file_handler, stream_handler])
    textbox_logger = logging.getLogger('textbox')
    textbox_logger.setLevel(log_level)

    if not enabled:
        textbox_logger.disabled = True


