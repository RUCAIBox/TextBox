# @Time   : 2020/11/4
# @Author : Gaole He
# @email  : hegaole@ruc.edu.cn

"""
textbox.data.dataloader.abstract_dataloader
################################################
"""

import math
import torch
from logging import getLogger

from textbox.utils import InputType
from textbox.utils.enum_type import SpecialTokens


class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data.
        And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Corpus): The corpus for partition of dataset.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.

    Attributes:
        dataset (dict): The necessary elements of this dataloader.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    dl_type = None

    def __init__(self, config, dataset,
                 batch_size=1, shuffle=False):
        self.config = config
        self.device = config['device']
        self.logger = getLogger()
        self.dataset = dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.pr = 0

        self.get_vocab(dataset)

        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS

    def get_vocab(self, dataset):
        if 'idx2token' in dataset:
            self.idx2token = dataset['idx2token']
            self.token2idx = dataset['token2idx']
        elif 'source_idx2token' in dataset:
            self.idx2token = dataset['source_idx2token']
            self.token2idx = dataset['source_token2idx']
        else:
            raise NotImplementedError

    def _build_data(self, text_data, token2idx, need_text_start_end=True):
        r"""transform text to id and add sos and eos token index.
        input:
            text_data: list -> list -> character, original text
            token2idx: dict, map token to index
            need_text_start_end, bool, indicates whether we should add sos and eos token index.
        output:
            text_idx_data: list -> list -> int, list of word index
            idx_length_data: list of sequence length
        """
        text_idx_data = []
        idx_length_data = []
        sos_token_idx = token2idx[self.sos_token]
        eos_token_idx = token2idx[self.eos_token]
        for text in text_data:
            if need_text_start_end:
                text_idx = [sos_token_idx] + self._token2idx(text, token2idx) + [eos_token_idx]
            else:
                text_idx = self._token2idx(text, token2idx)
            text_idx_data.append(text_idx)
            idx_length_data.append(len(text_idx))
        return text_idx_data, idx_length_data

    def _pad_batch_sequence(self, text_idx_data, idx_length_data):
        r"""padding a batch of word index data, to make them have equivalent length
        input:
            text_idx_data: list -> list -> int, a batch of word index
            idx_length_data: list -> int, a batch of sequence length
        output:
            new_data: torch.LongTensor (batch_size, max_length_in_batch)
            length: torch.LongTensor (batch_size)
        """
        # tp_batch_size = len(text_idx_data)
        # print(tp_batch_size)
        max_len = max(idx_length_data)
        new_data = []
        for seq, len_seq in zip(text_idx_data, idx_length_data):
            # print(seq, len_seq)
            if len_seq < max_len:
                new_data.append(seq + [self.padding_token_idx] * (max_len - len_seq))
            else:
                new_data.append(seq)
        new_data = torch.LongTensor(new_data)
        length = torch.LongTensor(idx_length_data)
        return new_data, length

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def _idx2token(self, inputs, idx2token):
        if isinstance(inputs, list):
            return [self._idx2token(x, idx2token) for x in inputs]
        return idx2token[inputs]

    def _token2idx(self, inputs, token2idx):
        if isinstance(inputs, list):
            return [self._token2idx(x, token2idx) for x in inputs]
        return token2idx.get(inputs, self.unknown_token_idx)

    def get_reference(self):
        r"""Get reference documents for current data loader
        return is supposed to be reference_corpus as list -> list -> word
        """
        raise NotImplementedError('Method [get_reference] should be implemented')

    def data_preprocess(self, dataset):
        r"""obtain necessary elements from dataset(dict) and conduct preprocess
        """
        raise NotImplementedError('Method [data_preprocess] should be implemented')

    @property
    def vocab_size(self):
        r"""The vocabulary size.
        """
        raise NotImplementedError('Method [vocab_size] should be implemented')

    @property
    def padding_token_idx(self):
        r"""The `int` index of the special token indicating the padding token.
        """
        return self.token2idx[self.padding_token]

    @property
    def unknown_token_idx(self):
        r"""The `int` index of the special token indicating the unknown token.
        """
        return self.token2idx[self.unknown_token]

    @property
    def sos_token_idx(self):
        r"""The `int` index of the special token indicating the start of sequence.
        """
        return self.token2idx[self.sos_token]

    @property
    def eos_token_idx(self):
        r"""The `int` index of the special token indicating the end of sequence.
        """
        return self.token2idx[self.eos_token]

    @property
    def pr_end(self):
        r"""This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        r"""Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        r"""Assemble next batch of data in form of Interaction, and return these data.
        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

    def set_batch_size(self, batch_size):
        r"""Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.logger.warning('Batch size is changed to {}'.format(batch_size))
