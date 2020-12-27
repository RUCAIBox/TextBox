# @Time   : 2020/11/16
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/26
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn

import numpy as np
import os
from logging import getLogger
from textbox.utils.enum_type import SpecialTokens


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path']
        self.logger = getLogger()

        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS
        self.mask_token = SpecialTokens.MASK

        self.max_vocab_size = config['max_vocab_size']
        self.max_seq_length = config['max_seq_length']

        restored_exist = self.detect_restored(self.dataset_path)
        if restored_exist:
            self._load_restored()
        else:
            self._from_scratch()

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.info('Loading data from scratch')

        self._get_preset()
        self._load_data(self.dataset_path)
        self._data_processing()
        self._dump_data(self.dataset_path)

    def _load_restored(self):
        """Load dataset from restored.
        Initialize attributes firstly, then load data from binary files.
        """
        self.logger.info('Loading data from restored')

        self._get_preset()
        self.load_restored(self.dataset_path)

    @staticmethod
    def check_file_exist(filename):
        if not os.path.isfile(filename):
            return False
        return True

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        raise NotImplementedError('Method [_get_preset] should be implemented.')

    def _load_data(self, dataset_path):
        r"""Load dataset with dataset split strategy.
        Args:
            dataset_path (str): path of dataset dir.
        """
        raise NotImplementedError('Method [_load_data] should be implemented.')

    def _dump_data(self, dataset_path):
        r"""dump dataset with processed dataset.
        Args:
            dataset_path (str): path of dataset dir.
        """
        raise NotImplementedError('Method [_dump_data] should be implemented.')

    def _text2id(self, text_data, token2idx):
        r"""transform text to id. but out of vocab word will still be saved as original word
        input:
            text_data: list -> list -> word, original text
            token2idx: dict, map token to index
        output:
            text_idx_data: list -> list -> int, list of word index
        """
        text_idx_data = []
        for text in text_data:
            text_idx = self._token2idx(text, token2idx)
            text_idx_data.append(text_idx)
        return text_idx_data

    def _id2text(self, idx_data, idx2token):
        r"""transform id to text.
        input:
            idx_data: list -> list -> int, list of word idx
            idx2token: dict, map token to index
        output:
            text_data: list -> list -> word, list of word
        """
        text_data = []
        for text in idx_data:
            text = self._idx2token(text, idx2token)
            text_data.append(text)
        return text_data

    def _token2idx(self, inputs, token2idx):
        if isinstance(inputs, list):
            return [self._token2idx(x, token2idx) for x in inputs]
        return token2idx.get(inputs, inputs)

    def _idx2token(self, inputs, idx2token):
        if isinstance(inputs, list):
            return [self._idx2token(x, idx2token) for x in inputs]
        return idx2token.get(inputs, inputs)

    def _data_processing(self):
        r"""Necessary processing steps for dataset.
        """
        raise NotImplementedError('Method [_data_processing] should be implemented.')

    def _build_vocab(self):
        r"""Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [_build_vocab] should be implemented.')

    def shuffle(self):
        r"""Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    @property
    def vocab_size(self):
        r"""The vocabulary size.
        """
        return len(self.token2idx)

    @property
    def padding_token_id(self):
        r"""The `int` index of the special token indicating the padding token.
        """
        return self.token2idx[self.padding_token]

    @property
    def unknown_token_id(self):
        r"""The `int` index of the special token indicating the unknown token.
        """
        return self.token2idx[self.unknown_token]

    @property
    def sos_token_id(self):
        r"""The `int` index of the special token indicating the start of sequence.
        """
        return self.token2idx[self.sos_token]

    @property
    def eos_token_id(self):
        r"""The `int` index of the special token indicating the end of sequence.
        """
        return self.token2idx[self.eos_token]

    @property
    def mask_token_id(self):
        r"""The `int` index of the special token indicating the mask token.
        """
        return self.token2idx[self.mask_token]

    @staticmethod
    def _calcu_split_ids(tot, ratios):
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

    def detect_restored(self, dataset_path):
        r"""Detect whether restored datasets exisit in dataset_path.
        """
        raise NotImplementedError('Method [detect_restored] should be implemented.')

    def split_by_ratio(self, ratios):
        r"""Split dataset by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.

        Returns:
            list: List of : `list -> int`, whose interaction features has been splitted.

        Note:
            Other than the first one, each part is rounded down.
        """
        pass

    def build(self):
        r"""Prepare splitted data elements for dataloader.

        Returns:
            list: List of dict : provide necessary elements for dataloader.
        """
        raise NotImplementedError('Method [build] should be implemented.')
