import torch
from logging import getLogger
from accelerate import Accelerator
from accelerate.utils import set_seed
from textbox.utils.logger import init_logger
from textbox.utils.utils import get_model, get_tokenizer, get_trainer, init_seed
from textbox.config.configurator import Config
from textbox.data.utils import data_preparation
from textbox.utils.dashboard import init_dashboard, start_dashboard, finish_dashboard

from typing import Optional


def run_textbox(
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        config_file_list: Optional[list] = None,
        config_dict: Optional[dict] = None
):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
    """
    from textbox.quick_start.experiment import Experiment
    experiment = Experiment(model, dataset, config_file_list, config_dict)
    experiment.run()


def run_multi_seed(
        seed_num: int,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        config_file_list: Optional[list] = None,
        config_dict: Optional[dict] = None,
):
    pass


