# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# @Time   : 2020/12/4
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

"""
textbox.data.utils
########################
"""

import copy
import os
import importlib

# from textbox.config import EvalSetting
from textbox.data.dataloader import *


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    # TODO: config can not load task_type
    task_type = config['task_type'].lower()
    if task_type == "unconditional":
        from .dataset import SingleSentenceDataset
        return SingleSentenceDataset(config)
    elif task_type == "translation" or task_type == "summarization":
        from .dataset import PairedSentenceDataset
        return PairedSentenceDataset(config)
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
    dataset = create_dataset(config)

    builded_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = builded_datasets
    phases = ['train', 'valid', 'test']

    if save:
        save_datasets(config['checkpoint_dir'], name=phases, dataset=builded_datasets)

    train_data = dataloader_construct(
        name='train',
        config=config,
        dataset=train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True
    )

    valid_data, test_data = dataloader_construct(
        name='evaluation',
        config=config,
        dataset=[valid_dataset, test_dataset],
        batch_size=config['eval_batch_size']
    )

    return train_data, valid_data, test_data


def dataloader_construct(name, config, dataset, batch_size=1, shuffle=False):
    """Get a correct dataloader class by calling :func:`get_data_loader` to construct dataloader.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset or list of Dataset): The split dataset for constructing dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Returns:
        AbstractDataLoader or list of AbstractDataLoader: Constructed dataloader in split dataset.
    """
    if not isinstance(dataset, list):
        dataset = [dataset]

    if not isinstance(batch_size, list):
        batch_size = [batch_size] * len(dataset)

    if len(dataset) != len(batch_size):
        raise ValueError('dataset {} and batch_size {} should have the same length'.format(dataset, batch_size))

    # model_type = config['MODEL_TYPE']
    # TODO: config can not load task_type
    task_type = config['task_type'].lower()
    logger = getLogger()
    logger.info('Build [{}] DataLoader for [{}]'.format(task_type, name))
    # logger.info(eval_setting)
    logger.info('batch_size = [{}], shuffle = [{}]\n'.format(batch_size, shuffle))

    DataLoader = get_data_loader(config)

    # try:
    ret = [
        DataLoader(
            config=config,
            dataset=ds,
            batch_size=bs,
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


def get_data_loader(config):
    """Return a dataloader class according to :attr:`config` and :attr:`eval_setting`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    # TODO: config can not load task_type
    task_type = config['task_type'].lower()
    if task_type == "unconditional":
        return SingleSentenceDataLoader
    elif task_type == "translation" or task_type == "summarization":
        return PairedSentenceDataLoader
    else:
        raise NotImplementedError("No such data loader for TASK_TYPE: {}".format(task_type))


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
