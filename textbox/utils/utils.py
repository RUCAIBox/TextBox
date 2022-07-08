import os
import datetime
import importlib
import random
import torch
import numpy as np

from textbox.utils.enum_type import PLM_MODELS


def get_local_time() -> str:
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%Y-%b-%d_%H-%M-%S')

    return cur


def ensure_dir(dir_path: str):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    os.makedirs(dir_path, exist_ok=True)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Generator: model class
    """
    try:
        model_name = 'Pretrained_Models' if model_name.lower() in PLM_MODELS else model_name
        model_file_name = model_name.lower()
        module_path = '.'.join(['...model', model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
        model_class = getattr(model_module, model_name)
    except:
        raise NotImplementedError("{} can't be found".format(model_name))
    return model_class


def get_trainer(model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_name (str): model name

    Returns:
        ~textbox.trainer.trainer.Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('textbox.trainer.trainer'), model_name + 'Trainer')
    except AttributeError:
        return getattr(importlib.import_module('textbox.trainer.trainer'), 'Trainer')


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
