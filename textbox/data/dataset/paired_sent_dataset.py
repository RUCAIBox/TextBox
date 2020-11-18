# @Time   : 2020/11/16
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

import os
import nltk
import collections
import random
import numpy as np
from logging import getLogger
from textbox.data.dataset import Dataset


class PairedSentenceDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset)
        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length']

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        self.token2idx = {}
        self.idx2token = {}
        self.source_text_data = []
        self.target_text_data = []

    def __len__(self):
        return sum([len(data) for data in self.source_text_data])

    def _load_data(self, dataset_path):
        """Load features.
        Firstly load interaction features, then user/item features optionally,
        finally load additional features if ``config['additional_feat_suffix']`` is set.
        Args:
            dataset_name (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        dataset_file = os.path.join(dataset_path, 'corpus_large.txt')
        if not os.path.isfile(dataset_file):
            raise ValueError('File {} not exist'.format(dataset_file))

        fin = open(dataset_file, 'r')
        for line in fin:
            source, target = line.strip().split('\t')
            self.source_text_data.append(nltk.word_tokenizer(source)[:self.max_source_length])
            self.target_text_data.append(nltk.word_tokenizer(target)[:self.max_target_length])
        fin.close()

    def _data_processing(self):
        self._build_vocab()

    def _build_vocab_from_text(self, text_data):
        word_list = list()
        for text in text_data:
            word_list.extend(text)
        tokens = [token for token, _ in collections.Counter(word_list).items()]
        tokens = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token] + tokens
        tokens = tokens[:self.max_vocab_size]
        idx2token = dict(zip(range(self.max_vocab_size), tokens))
        token2idx = dict(zip(tokens, range(self.max_vocab_size)))
        return idx2token, token2idx

    def _build_vocab(self):
        self.idx2token, self.token2idx = self._build_vocab_from_text(self.source_text_data + self.target_text_data)

    def shuffle(self):
        temp = list(zip(self.source_text_data, self.target_text_data))
        random.shuffle(temp)
        self.source_text_data[:], self.target_text_data[:] = zip(*temp)

    def split_by_ratio(self, ratios):
        """Split dataset by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.

        Returns:
            list: List of : `list -> int`, whose interaction features has been splitted.

        Note:
            Other than the first one, each part is rounded down.
        """
        self.logger.debug('split by ratios [{}]'.format(ratios))
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        tot_cnt = self.__len__()
        split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
        corpus_list = []
        for start, end in zip([0] + split_ids, split_ids + [tot_cnt]):
            src_text_data = self.source_text_data[start: end]
            tgt_text_data = self.target_text_data[start: end]
            tp_data = {
                'idx2token': self.idx2token,
                'token2idx': self.token2idx,
                'source_text_data': src_text_data,
                'target_text_data': tgt_text_data,
            }
            corpus_list.append(tp_data)
        return corpus_list

    def build(self, eval_setting=None):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~textbox.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~textbox.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of builded :class:`Dataset`.
        """
        self.shuffle()
        split_args = {'strategy': 'by_ratio', 'ratios': [0.8, 0.1, 0.1]}
        if split_args['strategy'] == 'by_ratio':
            corpus_list = self.split_by_ratio(split_args['ratios'])
        else:
            raise NotImplementedError()
        return corpus_list
