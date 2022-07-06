# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/9, 2021/1/29, 2020/12/04
# @Author : Tianyi Tang, Gaole He2021/10/10
# @Email  : steven_tang@ruc.edu.cn, hegaole@ruc.edu.cn

"""
textbox.data.utils
########################
"""

import os
from posix import listdir
import nltk
import collections
import torch
import copy
import shutil
from logging import getLogger
from textbox.utils.enum_type import SpecialTokens


def get_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    task_type = config['task_type'].lower()
    if task_type == "unconditional":
        from .dataset import SingleSentenceDataset
        return SingleSentenceDataset
    elif task_type == "attribute":
        from .dataset import AttributedSentenceDataset
        return AttributedSentenceDataset
    elif task_type in ["translation", "summarization"]:
        if config['model'] == 'PointerNet':
            from .dataset import CopyPairedSentenceDataset
            return CopyPairedSentenceDataset
        from .dataset import PairedSentenceDataset
        return PairedSentenceDataset

    elif task_type in ["kg2text"]:
        from .dataset import KGSentenceDataset
        return KGSentenceDataset
    elif config['dataset'] == 'WikiBio':
        from .dataset import WikiBioSentenceDataset
        return WikiBioSentenceDataset
    elif config['dataset'] == 'RotoWire':
        from .dataset import RotoWireSentenceDataset
        return RotoWireSentenceDataset
    else:
        raise NotImplementedError("No such dataset for TASK_TYPE: {}".format(task_type))


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

    task_type = config['task_type'].lower()
    logger = getLogger()
    logger.info('Build [{}] DataLoader for [{}]'.format(task_type, name))
    logger.info('batch_size = [{}], shuffle = [{}], drop_last = [{}]\n'.format(batch_size, shuffle, drop_last))

    DataLoader = get_dataloader(config)

    return DataLoader(
        config=config, dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, DDP=DDP
    )


def get_dataloader(config):
    """Return a dataloader class according to :attr:`config` and :attr:`split_strategy`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`split_strategy`.
    """
    task_type = config['task_type'].lower()
    if task_type == "unconditional":
        from .dataloader import SingleSentenceDataLoader
        return SingleSentenceDataLoader
    elif task_type == "attribute":
        from .dataloader import AttributedSentenceDataLoader
        return AttributedSentenceDataLoader
    elif task_type in ["translation", "summarization"]:
        if config['model'] == 'PointerNet':
            from .dataloader import CopyPairedSentenceDataLoader
            return CopyPairedSentenceDataLoader
        from .dataloader import PairedSentenceDataLoader
        return PairedSentenceDataLoader
    elif task_type in ["kg2text"]:
        from .dataloader import KGSentenceDataLoader
        return KGSentenceDataLoader
    elif config['dataset'] == 'WikiBio':
        from .dataloader import WikiBioSentenceDataLoader
        return WikiBioSentenceDataLoader
    elif config['dataset'] == 'RotoWire':
        from .dataloader import RotoWireSentenceDataLoader
        return RotoWireSentenceDataLoader
    else:
        raise NotImplementedError("No such dataloader for TASK_TYPE: {}".format(task_type))


def construct_quick_test_dataset(dataset_path):
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
                for line in fin.readlines()[:10]:
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


def tokenize(text, tokenize_strategy, language, multi_sentence):
    """Tokenize text data.

    Args:
        text (str): text data.
        tokenize_strategy (str): strategy of tokenizer.
        language (str): language of text.
        multi_sentence (bool): whether to split text into sentence level.
    
    Returns:
        List[str]: the tokenized text data.
    """
    if multi_sentence:
        text = text.split('\t')
        if tokenize_strategy == 'none':
            words = text
        elif tokenize_strategy == 'by_space':
            words = [t.split() for t in text]
        elif tokenize_strategy == 'nltk':
            words = [nltk.word_tokenize(t, language=language) for t in text]
    else:
        text.replace('\t', ' ')
        if tokenize_strategy == 'none':
            words = text
        elif tokenize_strategy == 'by_space':
            words = text.split()
        elif tokenize_strategy == 'nltk':
            words = nltk.word_tokenize(text, language=language)
    return words


def load_data(dataset_path, tokenize_strategy, max_length, language, multi_sentence, max_num):
    """Load dataset from split (train, valid, test).
    This is designed for single sentence format.

    Args:
        dataset_path (str): path of dataset dir.
        tokenize_strategy (str): strategy of tokenizer.
        max_length (int): max length of sequence.
        language (str): language of text.
        multi_sentence (bool): whether to split text into sentence level.
        max_num (int): max number of sequence.
    
    Returns:
        List[List[str]]: the text list loaded from dataset path.
    """
    if not os.path.isfile(dataset_path):
        raise ValueError('File {} not exist'.format(dataset_path))

    text = []
    with open(dataset_path, "r") as fin:
        for line in fin:
            line = line.strip().lower()
            words = tokenize(line, tokenize_strategy, language, multi_sentence)
            if isinstance(words, str):  # no split
                text.append(words)
            elif isinstance(words[0], str):  # single sentence
                text.append(words[:max_length])
            else:  # multiple sentences
                text.append([word[:max_length] for word in words[:max_num]])
    return text


def build_vocab(text, max_vocab_size, special_token_list):
    """Build vocabulary of list of text data.

    Args:
        text (List[List[List[str]]] or List[List[List[List[str]]]]): list of text data, consisting of multiple groups.
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
        if isinstance(group[0], str):  # single word
            word_list.extend(group)
        else:
            for doc in group:
                if isinstance(doc[0], str):  # single sentence
                    word_list.extend(doc)
                else:
                    for sent in doc:  
                        if isinstance(sent, tuple): # kg
                            word_list.extend(sent[0] + [sent[1]] + sent[2])
                        else: # multiple sentences
                            word_list.extend(sent)

    token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
    token_count.sort(reverse=True)
    tokens = [word for count, word in token_count]
    tokens = special_token_list + tokens
    tokens = tokens[:max_vocab_size]

    max_vocab_size = len(tokens)
    idx2token = dict(zip(range(max_vocab_size), tokens))
    token2idx = dict(zip(tokens, range(max_vocab_size)))

    return idx2token, token2idx, max_vocab_size

def build_attribute_vocab(text):
    """Build attribute vocabulary of list of attribute data.

    Args:
        text (List[List[List[str]]] or List[List[List[List[str]]]]): list of attribute data, consisting of multiple groups.
    
    Returns:
        tuple:
            - idx2token (dict): map index to token.
            - token2idx (dict): map token to index.
    """
    attribute_num = len(text[0][0]) if isinstance(text[0][0][0], str) else len(text[0][0][0])
    attribute_set = [set() for _ in range(attribute_num)]
    for group in text:
        for doc in group:
            if isinstance(doc[0], str):
                assert len(doc) == attribute_num
                for i, attr in enumerate(doc):
                    attribute_set[i].add(attr)
            else:
                for sent in doc:
                    assert len(sent) == attribute_num
                    for i, attr in enumerate(sent):
                        attribute_set[i].add(attr)

    idx2token = []
    token2idx = []
    for i in range(attribute_num):
        attribute = list(attribute_set[i])
        attribute_size = len(attribute)
        idx2token.append(dict(zip(range(attribute_size), attribute)))
        token2idx.append(dict(zip(attribute, range(attribute_size))))
    return idx2token, token2idx

def text2idx(text, token2idx, tokenize_strategy):
    r"""transform text to id and add sos and eos token index.

    Args:
        text (List[List[List[str]]] or List[List[List[List[str]]]]): list of text data, consisting of multiple groups.
        token2idx (dict): map token to index
        tokenize_strategy (str): strategy of tokenizer.
    
    Returns:
        idx (List[List[List[int]]] or List[List[List[List[int]]]]): word index
        length (List[List[int]] or List[List[List[int]]]): sequence length
        num (None or List[List[int]]): sequence number
    """
    new_idx = []
    new_length = []
    requires_start_end = tokenize_strategy != 'none'
    sos_idx = token2idx[SpecialTokens.SOS]
    eos_idx = token2idx[SpecialTokens.EOS]
    unknown_idx = token2idx[SpecialTokens.UNK]

    if isinstance(text[0][0][0], str):  # single sentence
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
        return new_idx, new_length, None
    else:  # multiple sentences
        new_num = []
        for group in text:
            idx = []
            length = []
            num = []
            for doc in group:
                doc_idx = []
                doc_length = []
                for sent in doc:
                    sent_idx = [token2idx.get(word, unknown_idx) for word in sent]
                    if requires_start_end:
                        sent_idx = [sos_idx] + sent_idx + [eos_idx]
                    doc_idx.append(sent_idx)
                    doc_length.append(len(sent_idx))
                idx.append(doc_idx)
                length.append(doc_length)
                num.append(len(doc))
            new_idx.append(idx)
            new_length.append(length)
            new_num.append(num)
        return new_idx, new_length, new_num

def attribute2idx(text, token2idx):
    r"""transform attribute to id.

    Args:
        text (List[List[List[str]]] or List[List[List[List[str]]]]): list of attribute data, consisting of multiple groups.
        token2idx (dict): map token to index

    Returns:
        idx (List[List[List[int]]] or List[List[List[List[int]]]]): attribute index
        length (None or List[List[int]]): sequence length
    """
    new_idx = []
    new_length = []
    for group in text:
        idx = []
        length = []
        for doc in group:
            if isinstance(doc[0], str):
                doc_idx = [token2idx[i][attr] for i, attr in enumerate(doc)]
            else:
                doc_idx = []
                for sent in doc:
                    sent_idx = [token2idx[i][attr] for i, attr in enumerate(sent)]
                    doc_idx.append(sent_idx)
                length.append(len(doc))
            idx.append(doc_idx)
        new_idx.append(idx)
        new_length.append(length)
    
    if new_length[0] != []:
        return new_idx, new_length
    else:
        return new_idx

def pad_sequence(idx, length, padding_idx, num=None):
    r"""padding a batch of word index data, to make them have equivalent length

    Args:
        idx (List[List[int]] or List[List[List[int]]]): word index
        length (List[int] or List[List[int]]): sequence length
        padding_idx (int): the index of padding token
        num (List[int]): sequence number
    
    Returns:
        idx (List[List[int]] or List[List[List[int]]]): word index
        length (List[int] or List[List[int]]): sequence length
        num (List[int]): sequence number
    """
    if num is None:
        max_length = max(length)
        new_idx = []
        for sent_idx, sent_length in zip(idx, length):
            new_idx.append(sent_idx + [padding_idx] * (max_length - sent_length))
        new_idx = torch.LongTensor(new_idx)
        length = torch.LongTensor(length)
        return new_idx, length, None
    else:
        max_length = max([max(sent_length) for sent_length in length])
        max_num = max(num)
        new_length = []
        new_idx = []
        for doc_idx, doc_length, doc_num in zip(idx, length, num):
            new_length.append(doc_length + [0] * (max_num - doc_num))
            new_sent_idx = []
            for sent_idx, sent_length in zip(doc_idx, doc_length):
                new_sent_idx.append(sent_idx + [padding_idx] * (max_length - sent_length))
            for _ in range(max_num - doc_num):
                new_sent_idx.append([0] * max_length)
            new_idx.append(new_sent_idx)

        new_num = torch.LongTensor(num)
        new_length = torch.LongTensor(new_length)
        new_idx = torch.LongTensor(new_idx)
        return new_idx, new_length, new_num
