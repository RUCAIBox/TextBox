# -*- coding: utf-8 -*-
# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
textbox.utils.utils
################################
"""

import os
import datetime
import importlib
import random
import torch
import numpy as np
from textbox.utils.enum_type import ModelType, PLM_MODELS
import time


class Timer:

    def __enter__(self):
        self.__stime = time.time()
        return self

    def __exit__(self, *exc_info):
        self.__etime = time.time()

    @property
    def duration(self):
        return self.__etime - self.__stime


def greater(x, y):
    return x > y


def less(x, y):
    return x < y


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
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
    model_submodule = ['GAN', 'LM', 'VAE', 'Seq2Seq', 'Attribute', 'Kb2Text']
    try:
        model_name = 'Transformers' if model_name.lower() in PLM_MODELS else model_name
        model_file_name = model_name.lower()
        for submodule in model_submodule:
            module_path = '.'.join(['...model', submodule, model_file_name])
            if importlib.util.find_spec(module_path, __name__):
                model_module = importlib.import_module(module_path, __name__)

        model_class = getattr(model_module, model_name)
    except:
        raise NotImplementedError("{} can't be found".format(model_file_name))
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (~textbox.utils.enum_type.ModelType): model type
        model_name (str): model name

    Returns:
        ~textbox.trainer.trainer.Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('textbox.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type in [ModelType.UNCONDITIONAL]:
            return getattr(importlib.import_module('textbox.trainer'), 'Trainer')
        elif model_type == ModelType.GAN:
            return getattr(importlib.import_module('textbox.trainer'), 'GANTrainer')
        elif model_type in [ModelType.SEQ2SEQ, ModelType.ATTRIBUTE]:
            return getattr(importlib.import_module('textbox.trainer'), 'Seq2SeqTrainer')
        else:
            return getattr(importlib.import_module('textbox.trainer'), 'Trainer')


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
