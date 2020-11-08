# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

import os
import nltk
import collections
import random
import numpy as np
from logging import getLogger
from textbox.data.corpus import Corpus
from textbox.utils.enum_type import SpecialTokens


class Dataset(object):
    def __init__(self, config, saved_dataset=None):
        self.config = config
        self.dataset_path = config['data_path']
        self.logger = getLogger()

        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS

        self.max_vocab_size = config['max_vocab_size']
        self.max_seq_length = config['max_seq_length']

        if saved_dataset is None:
            self._from_scratch()
        else:
            self._restore_saved_dataset(saved_dataset)

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.debug('Loading {} from scratch'.format(self.__class__))

        self._get_preset()
        self._load_data(self.dataset_path)
        self._data_processing()

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        raise NotImplementedError('Method [_get_preset] should be implemented.')

    # def _restore_saved_dataset(self, saved_dataset):
    #     """Restore saved dataset from ``saved_dataset``.
    #     Args:
    #         saved_dataset (str): path for the saved dataset.
    #     """
    #     self.logger.debug('Restoring dataset from [{}]'.format(saved_dataset))
    #
    #     if (saved_dataset is None) or (not os.path.isdir(saved_dataset)):
    #         raise ValueError('filepath [{}] need to be a dir'.format(saved_dataset))
    #
    #     with open(os.path.join(saved_dataset, 'basic-info.json')) as file:
    #         basic_info = json.load(file)
    #
    #     for k in basic_info:
    #         setattr(self, k, basic_info[k])
    #
    #     feats = ['inter', 'user', 'item']
    #     for name in feats:
    #         cur_file_name = os.path.join(saved_dataset, '{}.csv'.format(name))
    #         if os.path.isfile(cur_file_name):
    #             df = pd.read_csv(cur_file_name)
    #             setattr(self, '{}_feat'.format(name), df)
    #         else:
    #             setattr(self, '{}_feat'.format(name), None)
    #
    #     self._get_field_from_config()

    def _load_data(self, dataset_path):
        """Load dataset from file.
        """
        raise NotImplementedError('Method [_load_data] should be implemented.')

    def _data_processing(self):
        raise NotImplementedError('Method [_data_processing] should be implemented.')

    def _build_vocab(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
                """
        raise NotImplementedError('Method [_build_vocab] should be implemented.')

    def shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
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

    @staticmethod
    def _calcu_split_ids(tot, ratios):
        """Given split ratios, and total number, calculate the number of each part after splitting.

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

    def split_by_ratio(self, ratios):
        """Split dataset by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.

        Returns:
            list: List of : `list -> int`, whose interaction features has been splitted.

        Note:
            Other than the first one, each part is rounded down.
        """
        pass

    def build(self, eval_setting=None):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~textbox.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~textbox.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of builded :class:`Dataset`.
        """
        # ordering_args = eval_setting.ordering_args
        # if ordering_args['strategy'] == 'shuffle':
        #     self.shuffle()
        # elif ordering_args['strategy'] == 'by':
        #     self.sort(by=ordering_args['field'], ascending=ordering_args['ascending'])
        self.shuffle()

        # group_field = eval_setting.group_field

        # split_args = eval_setting.split_args
        split_args = {'strategy': 'by_ratio', 'ratios': [0.8, 0.1, 0.1]}
        if split_args['strategy'] == 'by_ratio':
            corpus_list = self.split_by_ratio(split_args['ratios'])
        else:
            raise NotImplementedError()
        return corpus_list
