# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/1/29, 2020/12/04
# @Author : Tianyi Tang, Gaole He
# @Email  : steven_tang@ruc.edu.cn, hegaole@ruc.edu.cn

"""
textbox.data.utils
########################
"""

import os
import nltk
import collections
import pickle

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
    elif task_type == "attribute":
        from .dataset import AttributedSentenceDataset
        return AttributedSentenceDataset(config)
    elif task_type in ["translation", "summarization"]:
        from .dataset import PairedSentenceDataset
        return PairedSentenceDataset(config)
    elif task_type in ["multi_dialog", "poem"]:
        from .dataset import MultipleSentenceDataset
        return MultipleSentenceDataset(config)
    else:
        raise NotImplementedError("No such dataset for TASK_TYPE: {}".format(task_type))


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

    train_data = dataloader_construct(
        name='train',
        config=config,
        dataset=train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        DDP=True
    )

    valid_data = dataloader_construct(
        name='valid',
        config=config,
        dataset=valid_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        DDP=True
    )

    test_data = dataloader_construct(
        name='test',
        config=config,
        dataset=test_dataset,
        batch_size=config['eval_batch_size'],
        drop_last=False,
    )

    return train_data, valid_data, test_data


def dataloader_construct(name, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
    """Get a correct dataloader class by calling :func:`get_data_loader` to construct dataloader.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset or list of Dataset): The split dataset for constructing dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
        drop_last (bool, optional): Whether the dataloader will drop the last batch. Defaults to ``True``.
        DDP (bool, optional): Whether the dataloader will distribute in different GPU. Defaults to ``False``.

    Returns:
        AbstractDataLoader or list of AbstractDataLoader: Constructed dataloader in split dataset.
    """

    task_type = config['task_type'].lower()
    logger = getLogger()
    logger.info('Build [{}] DataLoader for [{}]'.format(task_type, name))
    logger.info('batch_size = [{}], shuffle = [{}], drop_last = [{}]\n'.format(batch_size, shuffle, drop_last))

    DataLoader = get_data_loader(config)

    return DataLoader(
        config=config, dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, DDP=DDP
    )


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
    elif task_type == "attribute":
        return AttributedSentenceDataLoader
    elif task_type in ["translation", "summarization"]:
        return PairedSentenceDataLoader
    elif task_type in ["multi_dialog", "poem"]:
        return MultipleSentenceDataLoader
    else:
        raise NotImplementedError("No such data loader for TASK_TYPE: {}".format(task_type))


def tokenize(data, tokenize_strategy, language):
    """Tokenize text data.

    Args:
        data (str): text data.
        tokenize_strategy (str): strategy of tokenizer.
        language (str): language of text.
    
    Returns:
        List[str]: the tokenized text data.
    """
    if tokenize_strategy == 'by_space':
        words = data.split()
    else:
        words = nltk.word_tokenize(data, language=language)
    return words


def load_data(dataset_path, tokenize_strategy, overlength_strategy, max_seq_length, language):
    """Load dataset from split (train, dev, test).
    This is designed for single sentence format.

    Args:
        dataset_path (str): path of dataset dir.
        tokenize_strategy (str): strategy of tokenizer.
        overlength_strategy (str): strategy of overlengthed text.
        max_seq_length (int): max length of sequence.
        language (str): language of text.
    
    Returns:
        List[List[str]]: the text list loaded from dataset path.
    """
    if not os.path.isfile(dataset_path):
        raise ValueError('File {} not exist'.format(dataset_path))

    fin = open(dataset_path, "r")
    text = []
    for line in fin:
        line = line.strip().lower()
        words = tokenize(line, tokenize_strategy, language)

        if overlength_strategy == 'truncate':
            text.append(words[:max_seq_length])
        elif overlength_strategy == 'drop':
            if len(words) <= max_seq_length:
                text.append(words)
        elif overlength_strategy == 'none':
            text.append(words)
    fin.close()
    return text


def calcu_split_ids(tot, ratios):
    r"""Given split ratios, and total number, calculate the number of each part after splitting.

    Other than the first one, each part is rounded down.

    Args:
        tot (int): Total number.
        ratios (list): List of split ratios. No need to be normalized.

    Returns:
        list: Number of each part after splitting.
    """
    cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
    cnt[0] = tot - sum(cnt[1:])
    split_ids = np.cumsum(cnt)[:-1]
    return list(split_ids)


def split_data(data_list, ratios):
    """Split all the data in data list by ratios.

    Args:
        data_list (List): the data to be splitted.
        ratios (List[float, float, float]): spiltted ratios of train, dev, test dataset.

    Returns:
        List: the list of split data.
    """
    tot_ratio = sum(ratios)
    ratios = [_ / tot_ratio for _ in ratios]
    split_list = []

    for data in data_list:
        tot_cnt = len(data)
        split_ids = calcu_split_ids(tot_cnt, ratios)
        corpus_list = []
        for start, end in zip([0] + split_ids, split_ids + [tot_cnt]):
            corpus_list.append(data[start:end])
        split_list.append(corpus_list)

    return split_list


def build_vocab(text_data_list, max_vocab_size, special_token_list):
    """Build vocabulary of list of text data.

    Args:
        text_data_list (List[List[str]]): list of text data.
        max_vocab_size (int): max size of vocabulary.
        special_token_list (List[str]): list of special tokens.
    
    Returns:
        tuple:
            - idx2token (dict): map index to token.
            - token2idx (dict): map token to index.
            - max_vocab_size (int): updated max size of vocabulary.
    """

    word_list = list()
    for text_data in text_data_list:
        for text in text_data:
            if isinstance(text[0], str):
                word_list.extend(text)
            else:
                for words in text:
                    word_list.extend(words)

    token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
    token_count.sort(reverse=True)
    tokens = [word for count, word in token_count]
    tokens = special_token_list + tokens
    tokens = tokens[:max_vocab_size]

    max_vocab_size = len(tokens)
    idx2token = dict(zip(range(max_vocab_size), tokens))
    token2idx = dict(zip(tokens, range(max_vocab_size)))
    return idx2token, token2idx, max_vocab_size


def detect_restored(dataset_path, suffix="", ignore_file=""):
    """Detect whether binary files is already restored.

    Args:
        dataset_path (str): path of dataset dir.
        suffix (str, optional): suffix of files, default: "".
        ignore_file (str, optional): ignored file (data or vocab), default: "".
    
    Returns:
        bool: whether files are already restored.
    """
    required_files = []
    if ignore_file != "data":
        for prefix in ['train', 'dev', 'test']:
            filename = os.path.join(dataset_path, '{}.{}bin'.format(prefix, suffix))
            required_files.append(filename)
    if ignore_file != "vocab":
        vocab_file = os.path.join(dataset_path, '{}vocab'.format(suffix))
        required_files.append(vocab_file)
    absent_file_flag = False
    for filename in required_files:
        if not os.path.isfile(filename):
            absent_file_flag = True
            break
    return not absent_file_flag


def dump_data(dataset_path, text_data=None, idx2token=None, token2idx=None, suffix=""):
    """Dump data into binary files.

    Args:
        dataset_path (str): path of dataset dir.
        text_data (List[List[str]]): list of text data.
        idx2token (dict): map index to token.
        token2idx (dict): map token to index.
        suffix (str, optional): suffix of files, default: "".
    """
    if idx2token is not None:
        vocab_file = os.path.join(dataset_path, '{}vocab'.format(suffix))
        with open(vocab_file, "wb") as f_vocab:
            pickle.dump([idx2token, token2idx], f_vocab)

    if text_data is not None:
        for i, prefix in enumerate(['train', 'dev', 'test']):
            text = text_data[i]
            idx_filename = os.path.join(dataset_path, '{}.{}bin'.format(prefix, suffix))
            with open(idx_filename, "wb") as f_text:
                pickle.dump(text, f_text)


def load_restored(dataset_path, suffix="", ignore_file=""):
    """Load dataset from restored binary files (train, dev, test).

    Args:
        dataset_path (str): path of dataset dir.
        suffix (str, optional): suffix of files, default: "".
        ignore_file (str, optional): ignored file (data or vocab), default: "".
    
    Returns:
        tuple:
            - text_data (List[List[str]]): list of text data.
            - idx2token (dict, optional): map index to token.
            - token2idx (dict, optional): map token to index.
    """
    return_list = []
    if ignore_file != 'data':
        text_data = []
        for prefix in ['train', 'dev', 'test']:
            idx_filename = os.path.join(dataset_path, '{}.{}bin'.format(prefix, suffix))
            with open(idx_filename, "rb") as f_text:
                text = pickle.load(f_text)
            text_data.append(text)
        return_list.append(text_data)

    if ignore_file != 'vocab':
        vocab_file = os.path.join(dataset_path, '{}vocab'.format(suffix))
        with open(vocab_file, "rb") as f_vocab:
            idx2token, token2idx = pickle.load(f_vocab)
        return_list.extend([idx2token, token2idx])

    return return_list
