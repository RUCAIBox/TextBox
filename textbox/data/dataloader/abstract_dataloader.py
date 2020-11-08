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
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Corpus): The corpus for partition of dataset.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~textbox.utils.enum_type.InputType.SentencePair`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        real_time (bool): If ``True``, dataloader will do data pre-processing,
            such as neg-sampling and data-augmentation.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    dl_type = None

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.NOISE, shuffle=False):
        self.config = config
        self.device = config['device']
        self.logger = getLogger()
        self.dataset = dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.dl_format = dl_format
        self.shuffle = shuffle
        self.pr = 0

        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS

        self.setup()
        self.data_preprocess(dataset)

    def setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def get_reference(self):
        """Return target sequence as reference, mainly for evaluation.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _pad_batch_sequence(self, text_idx_data, idx_length_data):
        '''
        input:
            cur_data: list -> list -> int
        output:
            new_data: torch.LongTensor (batch_size, max_length_in_batch)
            length: torch.LongTensor (batch_size)
        '''
        # tp_batch_size = len(text_idx_data)
        # print(tp_batch_size)
        max_len = max(idx_length_data)
        new_data = []
        for seq, len_seq in zip(text_idx_data, idx_length_data):
            # print(seq, len_seq)
            if len_seq < max_len:
                new_data.append(seq + [self.padding_index] * (max_len - len_seq))
            else:
                new_data.append(seq)
        new_data = torch.LongTensor(new_data)
        length = torch.LongTensor(idx_length_data)
        return new_data, length

    def data_preprocess(self, dataset):
        """This function is used to do some data preprocess, such as pre-neg-sampling and pre-data-augmentation.
        By default, it will do nothing.
        """
        pass

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

    @property
    def vocab_size(self):
        r"""The vocabulary size.
        """
        return self.dataset.vocab_size

    @property
    def padding_token_idx(self):
        r"""The `int` index of the special token indicating the padding token.
        """
        return self.dataset.padding_token_idx

    @property
    def unknown_token_idx(self):
        r"""The `int` index of the special token indicating the unknown token.
        """
        return self.dataset.unknown_token_idx

    @property
    def sos_token_idx(self):
        r"""The `int` index of the special token indicating the start of sequence.
        """
        return self.dataset.sos_token_idx

    @property
    def eos_token_idx(self):
        r"""The `int` index of the special token indicating the end of sequence.
        """
        return self.dataset.eos_token_idx

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

    def set_batch_size(self, batch_size):
        """Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.logger.warning('Batch size is changed to {}'.format(batch_size))
