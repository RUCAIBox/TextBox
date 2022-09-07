import os
import datetime
import importlib
from logging import getLogger
from typing import Union, Optional

import torch
from accelerate.utils import set_seed
from transformers import AutoTokenizer, BertTokenizer#, BertTokenierForUnilm
from ..model.unilm_v1.tokenization_unilm import BertTokenizerForUnilm
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


def safe_remove(dir_path: Optional[str], overwrite: bool = True):
    """
    `safe_remove` is a function that removes a file or soft link at a given path, and if the file or soft link doesn't
    exist, it does nothing

    Args:
        dir_path: The path to the directory you want to remove
        overwrite: (default = True) If True, the file will be deleted.
            If False, the file will be renamed with the current time.
    """
    if not dir_path:
        return
    if os.path.exists(dir_path) or os.path.islink(dir_path):
        if overwrite:
            os.remove(dir_path)
        else:
            dir_path += get_local_time()


def get_tag(_tag: Optional[str], _serial: Optional[int]):
    r"""
    Get the file tag with serial number.

    Examples:
        >>> get_tag('epoch', 1)
        _epoch-1
    """
    _tag = '' if _tag is None else '_' + _tag
    if _serial is not None:
        _tag += '-' + str(_serial)
    return _tag


def serialized_save(
        source: Union[dict, list],
        serial: Optional[int],
        serial_of_soft_link: Optional[int],
        path_without_extension: str,
        tag: Optional[str] = None,
        extension_name: Optional[str] = None,
        overwrite: bool = True,
        max_save: int = -1,
):
    r"""
    Save a sequence of files with given serial numbers and create a soft link
    to the file with specified serial number.

    Args:
        source: The source of current file.
        serial: The serial number of current file.
        serial_of_soft_link: The serial number that the soft link will point to.
        path_without_extension: The path to base file without extension name.
            This should remain the same within the sequence.
        tag: The extended tag of filename like 'epoch' or 'valid'.
            This should remain the same within the sequence.
        extension_name: (default = None) The extension name of file. This can also
            be specific automatically if leave blank.
        overwrite: (default = True) Whether to overwrite the file to be saved.
        max_save: (default = -1) The maximal amount of files. If -1, every file
            will be saved. 1: only the file with serial number same as `serial_
            of_soft_link` will be saved. 2: both the last one and linked files.
    """
    if max_save == 0 or (max_save == 1 and serial_of_soft_link != serial):
        return

    # deal with naming
    if extension_name not in ('txt', 'pth'):
        extension_name = 'txt' if isinstance(source, list) else 'pth'
    path_to_save = os.path.abspath(path_without_extension + get_tag(tag, serial) + '.' + extension_name)
    safe_remove(path_to_save, overwrite)  # behavior of torch.save is not clearly defined.
    getLogger(__name__).debug(f'Saving file to {path_to_save}')

    path_to_link = os.path.abspath(path_without_extension + '.' + extension_name)
    path_to_pre_best = os.readlink(path_to_link) if os.path.exists(path_to_link) else ''
    if not os.path.exists(path_to_pre_best):
        path_to_pre_best = None

    # save
    if extension_name == 'txt':
        with open(path_to_save, 'w') as fout:
            for text in source:
                fout.write(text + '\n')
    else:
        torch.save(source, path_to_save)

    # delete the file before the max_save
    if max_save != -1 and 0 <= serial - max_save + 1 < serial:
        path_to_delete = os.path.abspath(
            path_without_extension + get_tag(tag, serial - max_save + 1) + '.' + extension_name
        )
        if not path_to_pre_best or not os.path.samefile(path_to_delete, path_to_pre_best):
            safe_remove(path_to_delete)

    # create soft link
    if serial_of_soft_link is not None and serial_of_soft_link == serial:
        safe_remove(path_to_pre_best)
        safe_remove(path_to_link)
        os.symlink(path_to_save, path_to_link)


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
        if (config['model_name'] in ['chinese-bart', 'chinese-pegasus', 'cpt']):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        elif config['model_name'] in ['unilm']:
            tokenizer = BertTokenizerForUnilm.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        else:
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
    set_seed(seed)

    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
