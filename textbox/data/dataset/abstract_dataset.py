# @Time   : 2020/11/16
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2021/1/29
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.abstract_dataset
#####################################
"""

import numpy as np
import os
from logging import getLogger
from textbox.utils.enum_type import SpecialTokens


class AbstractDataset(object):
    """:class:`AbstractDataset` is an abstract object which stores the original dataset in memory.
        And it is also the ancestor of all other dataset.

    Args:
        config (Config): Global configuration object.
    """

    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path']
        self.logger = getLogger()

        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS
        self.special_token_list = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token]
        if ('user_token_list' in config):
            self.user_token_list = config['user_token_list']
            self.user_token_idx = [4 + i for i, _ in enumerate(self.user_token_list)]
            self.special_token_list += self.user_token_list

        self.tokenize_strategy = config['tokenize_strategy']
        self.overlength_strategy = config['overlength_strategy']
        self.split_strategy = config['split_strategy']
        assert self.split_strategy is not None
        self.split_ratio = config['split_ratio']

        self.restored_exist = self._detect_restored(self.dataset_path)
        self._get_preset()
        if self.restored_exist:
            self._from_restored()
        else:
            self._from_scratch()

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        raise NotImplementedError('Method [_get_preset] should be implemented.')

    def _from_scratch(self):
        """Load dataset from scratch. Firstly load data from atomic files, then build vocabulary, dump data lastly.
        """
        self.logger.info('Loading data from scratch')

        self._load_data(self.dataset_path)
        self._build_vocab()
        self._dump_data(self.dataset_path)

    def _from_restored(self):
        """Load dataset from restored binary files.
        """
        self.logger.info('Loading data from restored')

        self._load_restored(self.dataset_path)

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

    def _build_vocab(self):
        r"""Build the vocabulary of text data.
        """
        raise NotImplementedError('Method [_build_vocab] should be implemented.')

    def _detect_restored(self, dataset_path):
        r"""Detect whether restored datasets exist in dataset_path.
        """
        raise NotImplementedError('Method [_detect_restored] should be implemented.')

    def build(self):
        r"""Prepare splitted data elements for dataloader.

        Returns:
            list: List of dict : provide necessary elements for dataloader.
        """
        raise NotImplementedError('Method [build] should be implemented.')

    @staticmethod
    def check_file_exist(filename):
        return os.path.isfile(filename)
