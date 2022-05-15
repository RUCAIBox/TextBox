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


log_dir = './log/'
file_fmt = "%(asctime)-15s %(levelname)s %(message)s"
file_date_fmt = "%a %d %b %Y %H:%M:%S"
stream_fmt = "%(asctime)-15s %(levelname)s %(message)s"
stream_date_fmt = "%d %b %H:%M"


def init_logger(config, is_logger: bool):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        is_logger (bool): Whether to log

    Example:
        >>> logger = logging.getLogger()
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    dir_name = os.path.dirname(log_dir)
    ensure_dir(dir_name)

    log_filename = config['filename'] + '.log'
    log_filepath = os.path.join(log_dir, log_filename)

    file_formatter = logging.Formatter(file_fmt, file_date_fmt)
    stream_formatter = logging.Formatter(stream_fmt, stream_date_fmt)

    if config['state'] is None:
        config['state'] = 'info'
    if not is_logger:
        config['state'] = 'critical'
    level = getattr(logging, config['state'].upper(), None)

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(stream_formatter)

    logging.basicConfig(level=level, handlers=[file_handler, stream_handler])
