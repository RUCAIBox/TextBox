import os
from posix import listdir
import nltk
import collections
import torch
import copy
import shutil
from logging import getLogger
from textbox import SpecialTokens

from typing import Any


def get_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    from .paired_sent_dataset import PairedSentenceDataset
    return PairedSentenceDataset


def get_dataloader(config):
    """Return a dataloader class according to :attr:`config` and :attr:`split_strategy`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`split_strategy`.
    """
    from .paired_sent_dataloader import PairedSentenceDataLoader
    return PairedSentenceDataLoader


def dataloader_construct(name, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
    """Get a correct dataloader class by calling :func:`get_dataloader` to construct dataloader.

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

    logger = getLogger()
    logger.info('Build DataLoader for [{}]'.format(name))
    logger.info('batch_size = [{}], shuffle = [{}], drop_last = [{}]\n'.format(batch_size, shuffle, drop_last))

    DataLoader = get_dataloader(config)

    return DataLoader(
        config=config, dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, DDP=DDP
    )


def construct_quick_test_dataset(dataset_path: str, num: int = 10):
    files = listdir(dataset_path)
    for file in files:
        filename = os.path.join(dataset_path, file)
        if filename.endswith('.bin'):
            os.remove(filename)
        else:
            shutil.copy(filename, filename + '.tmp')
    for file in files:
        filename = os.path.join(dataset_path, file)
        if not filename.endswith('.bin'):
            with open(filename + '.tmp', 'r') as fin, open(filename, 'w') as fout:
                for line in fin.readlines()[:num]:
                    fout.write(line)


def deconstruct_quick_test_dataset(dataset_path):
    files = listdir(dataset_path)
    for file in files:
        filename = os.path.join(dataset_path, file)
        if filename.endswith('.bin'):
            os.remove(filename)
        elif not filename.endswith('.tmp'):
            shutil.move(filename + '.tmp', filename)


def data_preparation(config, save=False):
    """call :func:`dataloader_construct` to create corresponding dataloader.

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
    if config['quick_test']:
        if isinstance(config['quick_test'], int):
            construct_quick_test_dataset(config['data_path'], num=config['quick_test'])
        else:
            construct_quick_test_dataset(config['data_path'])
    dataset = get_dataset(config)(config)
    if config['quick_test']:
        deconstruct_quick_test_dataset(config['data_path'])

    train_dataset = copy.copy(dataset)
    valid_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)
    for prefix in ['train', 'valid', 'test']:
        dataset = locals()[f'{prefix}_dataset']
        content = getattr(dataset, f'{prefix}_data')
        for key, value in content.items():
            setattr(dataset, key, value)

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


def tokenize(text, tokenize_strategy, language):
    """Tokenize text data.

    Args:
        text (str): text data.
        tokenize_strategy (str): strategy of tokenizer.
        language (str): language of text.
    
    Returns:
        List[str]: the tokenized text data.
    """
    text.replace('\t', ' ')
    if tokenize_strategy == 'none':
        words = text
    elif tokenize_strategy == 'by_space':
        words = text.split()
    elif tokenize_strategy == 'nltk':
        words = nltk.word_tokenize(text, language=language)
    return words


def load_data(dataset_path, tokenize_strategy, max_length, language):
    """Load dataset from split (train, valid, test).
    This is designed for single sentence format.

    Args:
        dataset_path (str): path of dataset dir.
        tokenize_strategy (str): strategy of tokenizer.
        max_length (int): max length of sequence.
        language (str): language of text.
    
    Returns:
        List[List[str]]: the text list loaded from dataset path.
    """
    if not os.path.isfile(dataset_path):
        raise ValueError('File {} not exist'.format(os.path.abspath(dataset_path)))

    text = []
    with open(dataset_path, "r") as fin:
        for line in fin:
            line = line.strip()
            if len(line) >= 2 and ((line[0] == '"' and line[-1] == '"') or (line[0] == "'" and line[-1] == "'")):
                try:
                    line = str(eval(line))
                except:
                    pass
            words = tokenize(line, tokenize_strategy, language)
            if isinstance(words, str):  # no split
                text.append(words)
            else: # single sentence
                text.append(words[:max_length])
    return text


def build_vocab(text, max_vocab_size, special_token_list):
    """Build vocabulary of list of text data.

    Args:
        text (List[List[List[str]]]): list of text data, consisting of multiple groups.
        max_vocab_size (int): max size of vocabulary.
        special_token_list (List[str]): list of special tokens.
    
    Returns:
        tuple:
            - idx2token (dict): map index to token.
            - token2idx (dict): map token to index.
            - max_vocab_size (int): updated max size of vocabulary.
    """

    word_list = list()
    for group in text:  # train, valid, test
        for doc in group:
            word_list.extend(doc)

    token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
    token_count.sort(reverse=True)
    tokens = [word for count, word in token_count]
    tokens = special_token_list + tokens
    tokens = tokens[:max_vocab_size]

    max_vocab_size = len(tokens)
    idx2token = dict(zip(range(max_vocab_size), tokens))
    token2idx = dict(zip(tokens, range(max_vocab_size)))

    return idx2token, token2idx, max_vocab_size

def text2idx(text, token2idx, tokenize_strategy):
    r"""transform text to id and add sos and eos token index.

    Args:
        text (List[List[List[str]]]): list of text data, consisting of multiple groups.
        token2idx (dict): map token to index
        tokenize_strategy (str): strategy of tokenizer.
    
    Returns:
        idx (List[List[List[int]]]): word index
        length (List[List[int]]): sequence length
    """
    new_idx = []
    new_length = []
    requires_start_end = tokenize_strategy != 'none'
    sos_idx = token2idx[SpecialTokens.BOS]
    eos_idx = token2idx[SpecialTokens.EOS]
    unknown_idx = token2idx[SpecialTokens.UNK]

    for group in text:
        idx = []
        length = []
        for sent in group:
            sent_idx = [token2idx.get(word, unknown_idx) for word in sent]
            if requires_start_end:
                sent_idx = [sos_idx] + sent_idx + [eos_idx]
            idx.append(sent_idx)
            length.append(len(sent_idx))
        new_idx.append(idx)
        new_length.append(length)
    return new_idx, new_length


def pad_sequence(idx, length, padding_idx):
    r"""padding a batch of word index data, to make them have equivalent length

    Args:
        idx (List[List[int]]): word index
        length (List[int]): sequence length
        padding_idx (int): the index of padding token
    
    Returns:
        idx (List[List[int]]): word index
        length (List[int]): sequence length
    """
    max_length = max(length)
    new_idx = []
    for sent_idx, sent_length in zip(idx, length):
        new_idx.append(sent_idx + [padding_idx] * (max_length - sent_length))
    new_idx = torch.LongTensor(new_idx)
    length = torch.LongTensor(length)
    return new_idx, length
