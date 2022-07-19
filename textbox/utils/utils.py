import os
import datetime
import importlib
import random
from typing import Union, Optional

import torch
import numpy as np
from transformers import AutoTokenizer

from .enum_type import PLM_MODELS


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


def serialized_save(
        source: Union[dict, list],
        path_without_extension: str,
        serial: Optional[int],
        serial_of_soft_link: Optional[int],
        tag: Optional[str] = None,
        overwrite: bool = True,
        save_method: Optional[str] = None,
):
    r"""Store the model parameters information and training information.

    Save checkpoint every validation as the formate of 'Model-Dataset-Time_epoch-?.pth'. A soft link named
    'Model-Dataset-Time.pth' pointing to the best epoch will be created.

    Todo:
        * Update docstring
        - maintain useless files
        - add save strategy
    """

    # deal with naming
    tag = '' if tag is None else '_' + tag
    if serial is not None:
        tag += '-' + str(serial)

    path_to_save = os.path.abspath(path_without_extension + tag)
    if os.path.exists(path_to_save):
        if overwrite:
            os.remove(path_to_save)  # behavior of torch.save is not clearly defined.
        else:
            path_to_save += get_local_time()

    # save
    if save_method is None or save_method not in ('txt', 'pth'):
        save_method = 'txt' if isinstance(source, list) else 'pth'
    path_to_save += save_method

    if save_method == 'txt':
        with open(path_to_save, 'w') as fout:
            for text in source:
                fout.write(text + '\n')
    else:
        torch.save(source, path_to_save)

    # create soft link to best model
    if serial_of_soft_link == serial:
        path_to_best = os.path.abspath(path_without_extension + save_method)
        if os.path.exists(path_to_best):
            os.remove(path_to_best)
        os.symlink(path_to_save, path_to_best)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Notes:
        model_name should be lowercase!

    Args:
        model_name (str): model name

    Returns:
        Generator: model class
    """
    if model_name.lower() in PLM_MODELS:
        model_name = 'Pretrained_Models'
    module_path = '.'.join(['...model', model_name.lower()])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
        model_class = getattr(model_module, model_name)
    else:
        raise ValueError("{} can't be found".format(model_name))
    return model_class


def get_trainer(model_name):
    r"""Automatically select trainer class based on model type and model name

    Notes:
        model_name should be original string (typically is upper case) like "BART"

    Args:
        model_name (str): model name

    Returns:
        ~textbox.trainer.trainer.Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('textbox.trainer.trainer'), model_name + 'Trainer')
    except AttributeError:
        return getattr(importlib.import_module('textbox.trainer.trainer'), 'Trainer')


def get_tokenizer(config):
    model_name = config['model_name']
    if model_name in PLM_MODELS:
        tokenizer_kwargs = config['tokenizer_kwargs'] or {}
        tokenizer_path = config['tokenizer_path'] or config['model_path']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

        # (1): tokenizer needs to add eos token
        if model_name in ['ctrl', 'openai-gpt']:
            tokenizer.add_special_tokens(({'eos_token': '</s>'}))

        # (2): tokenizer needs to add pad token
        if model_name in ['ctrl', 'gpt2', 'gpt_neo', 'openai-gpt']:
            tokenizer.pad_token = tokenizer.eos_token

        # (3): tokenizer needs to change replace eos token with sep token
        if model_name in ['cpm']:
            tokenizer.eos_token = tokenizer.sep_token

        # (4): tokenizer needs to modify `build_inputs_with_special_tokens()` and `num_special_tokens_to_add()`
        if model_name in ['blenderbot-small', 'cpm', 'ctrl', 'gpt2', 'gpt_neo', 'openai-gpt']:
            tokenizer.build_inputs_with_special_tokens = lambda t0, t1=None: t0 + [tokenizer.eos_token_id]
            tokenizer.num_special_tokens_to_add = lambda: 1
        elif model_name in ['opt']:
            tokenizer.build_inputs_with_special_tokens = \
                lambda t0, t1=None: [tokenizer.bos_token_id] + t0 + [tokenizer.eos_token_id]
            tokenizer.num_special_tokens_to_add = lambda: 2

        # (5): tokenizer needs to set src_lang, tgt_lang (used in translation task)
        if model_name in ['m2m_100', 'mbart']:
            assert config['src_lang'] and config['tgt_lang'], \
                model_name + ' needs to specify source language and target language ' \
                             'with `--src_lang=xx` and `--tgt_lang=xx`'
            tokenizer.src_lang = config['src_lang']
            tokenizer.tgt_lang = config['tgt_lang']

    return tokenizer


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
