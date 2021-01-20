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

from textbox.data.dataloader import *


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
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
    """Split the dataset by :attr:`config['split_strategy']` and call :func:`dataloader_construct` to create
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
        name='train', config=config, dataset=train_dataset, batch_size=config['train_batch_size'], shuffle=True
    )

    valid_data, test_data = dataloader_construct(
        name='evaluation', config=config, dataset=[valid_dataset, test_dataset], batch_size=config['eval_batch_size']
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

    task_type = config['task_type'].lower()
    logger = getLogger()
    logger.info('Build [{}] DataLoader for [{}]'.format(task_type, name))
    logger.info('batch_size = [{}], shuffle = [{}]\n'.format(batch_size, shuffle))

    DataLoader = get_data_loader(config)

    ret = [DataLoader(config=config, dataset=ds, batch_size=bs, shuffle=shuffle) for ds, bs in zip(dataset, batch_size)]

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


def get_data_loader(config):
    """Return a dataloader class according to :attr:`config` and :attr:`split_strategy`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`split_strategy`.
    """
    task_type = config['task_type'].lower()
    if task_type == "unconditional":
        return SingleSentenceDataLoader
    elif task_type == "translation" or task_type == "summarization":
        return PairedSentenceDataLoader
    else:
        raise NotImplementedError("No such data loader for TASK_TYPE: {}".format(task_type))
