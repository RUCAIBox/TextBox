# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/19, 2020/9/17, 2020/8/31
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com

"""
textbox.data.utils
########################
"""

import copy
import os
import importlib

from textbox.config import EvalSetting
from textbox.utils import ModelType
from textbox.data.dataloader import *


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    model_type = config['MODEL_TYPE']
    if model_type == ModelType.UNCONDITIONAL:
        from .dataset import SingleSentenceDataset
        return SingleSentenceDataset(config)
    elif model_type == ModelType.TRANSLATION or model_type == ModelType.CONDITIONAL:
        from .dataset import TranslationDataset
        return TranslationDataset(config)
    else:
        from .dataset import Dataset
        return Dataset(config)


def data_preparation(config, save=False):
    """Split the dataset by :attr:`config['eval_setting']` and call :func:`dataloader_construct` to create
    corresponding dataloader.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        save (bool, optional): If ``True``, it will call :func:`save_datasets` to save split dataset.
            Defaults to ``False``.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    model_type = config['MODEL_TYPE']

    if model_type == ModelType.UNCONDITIONAL:
        from .dataset import SingleSentenceDataset
        dataset = SingleSentenceDataset(config)
    elif model_type == ModelType.TRANSLATION or model_type == ModelType.CONDITIONAL:
        from .dataset import TranslationDataset
        dataset = TranslationDataset(config)
    else:
        raise NotImplementedError("model of type {} is not implemented".format(model_type))

    builded_datasets = dataset.build(eval_setting=None)
    train_dataset, valid_dataset, test_dataset = builded_datasets
    phases = ['train', 'valid', 'test']

    if save:
        save_datasets(config['checkpoint_dir'], name=phases, dataset=builded_datasets)

    train_data = dataloader_construct(
        name='train',
        config=config,
        eval_setting=None,
        dataset=train_dataset,
        dl_format=config['MODEL_INPUT_TYPE'],
        batch_size=config['train_batch_size'],
        shuffle=True
    )

    valid_data, test_data = dataloader_construct(
        name='evaluation',
        config=config,
        eval_setting=None,
        dataset=[valid_dataset, test_dataset],
        batch_size=config['eval_batch_size']
    )

    return train_data, valid_data, test_data


def dataloader_construct(name, config, eval_setting, dataset,
                         dl_format=InputType.NOISE,
                         batch_size=1, shuffle=False):
    """Get a correct dataloader class by calling :func:`get_data_loader` to construct dataloader.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.
        dataset (Dataset or list of Dataset): The split dataset for constructing dataloader.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~textbox.utils.enum_type.InputType.NOISE`.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
        **kwargs: Other input args of dataloader, such as :attr:`sampler`, :attr:`kg_sampler`
            and :attr:`neg_sample_args`. The meaning of these args is the same as these args in some dataloaders.

    Returns:
        AbstractDataLoader or list of AbstractDataLoader: Constructed dataloader in split dataset.
    """
    if not isinstance(dataset, list):
        dataset = [dataset]

    if not isinstance(batch_size, list):
        batch_size = [batch_size] * len(dataset)

    if len(dataset) != len(batch_size):
        raise ValueError('dataset {} and batch_size {} should have the same length'.format(dataset, batch_size))

    model_type = config['MODEL_TYPE']
    logger = getLogger()
    logger.info('Build [{}] DataLoader for [{}] with format [{}]'.format(model_type, name, dl_format))
    # logger.info(eval_setting)
    logger.info('batch_size = [{}], shuffle = [{}]\n'.format(batch_size, shuffle))

    DataLoader = get_data_loader(name, config, eval_setting)

    # try:
    ret = [
        DataLoader(
            config=config,
            dataset=ds,
            batch_size=bs,
            dl_format=dl_format,
            shuffle=shuffle
        ) for ds, bs in zip(dataset, batch_size)
    ]
    # except TypeError:
    #     raise ValueError('training_neg_sample_num should be 0')

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def save_datasets(save_path, name, dataset):
    """Save split datasets.

    Args:
        save_path (str): The path of directory for saving.
        name (str or list of str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        dataset (Dataset or list of Dataset): The split datasets.
    """
    if (not isinstance(name, list)) and (not isinstance(dataset, list)):
        name = [name]
        dataset = [dataset]
    if len(name) != len(dataset):
        raise ValueError('len of name {} should equal to len of dataset'.format(name, dataset))
    print("To be designed, nothing to save")
    # for i, d in enumerate(dataset):
    #     cur_path = os.path.join(save_path, name[i])
    #     if not os.path.isdir(cur_path):
    #         os.makedirs(cur_path)
    #     d.save(cur_path)


def get_data_loader(name, config, eval_setting):
    """Return a dataloader class according to :attr:`config` and :attr:`eval_setting`.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    model_type = config['MODEL_TYPE']
    if model_type == ModelType.UNCONDITIONAL:
        return SingleSentenceDataLoader
    elif model_type == ModelType.CONDITIONAL or model_type == ModelType.TRANSLATION:
        return TranslationDataLoader
    else:
        raise NotImplementedError("No such data loader for MODEL_TYPE: {}".format(model_type))
    # if model_type == ModelType.GENERAL or model_type == ModelType.TRADITIONAL:
    #     if neg_sample_strategy == 'none':
    #         return GeneralDataLoader
    #     elif neg_sample_strategy == 'by':
    #         return GeneralNegSampleDataLoader
    #     elif neg_sample_strategy == 'full':
    #         return GeneralFullDataLoader
    # else:
    #     raise NotImplementedError('model_type [{}] has not been implemented'.format(model_type))


class DLFriendlyAPI(object):
    """A Decorator class, which helps copying :class:`Dataset` methods to :class:`DataLoader`.

    These methods are called *DataLoader Friendly APIs*.

    E.g. if ``train_data`` is an object of :class:`DataLoader`,
    and :meth:`~recbole.data.dataset.dataset.Dataset.num` is a method of :class:`~recbole.data.dataset.dataset.Dataset`,
    Cause it has been decorated, :meth:`~recbole.data.dataset.dataset.Dataset.num` can be called directly by ``train_data``.

    See the example of :meth:`set` for details.

    Attributes:
        dataloader_apis (set): Register table that saves all the method names of DataLoader Friendly APIs.
    """
    def __init__(self):
        self.dataloader_apis = set()

    def __iter__(self):
        return self.dataloader_apis

    def set(self):
        """
        Example:
            .. code:: python

                from recbole.data.utils import dlapi

                @dlapi.set()
                def dataset_meth():
                    ...
        """
        def decorator(f):
            self.dataloader_apis.add(f.__name__)
            return f
        return decorator


dlapi = DLFriendlyAPI()
