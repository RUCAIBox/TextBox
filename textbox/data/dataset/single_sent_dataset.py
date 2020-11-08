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
from textbox.data.dataset import Dataset


class SingleSentenceDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset)

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        self.token2idx = {}
        self.idx2token = {}
        self.text_data = []

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

        fin = open(dataset_file, "r")
        for line in fin:
            words = nltk.word_tokenize(line.strip())[:self.max_seq_length]
            self.text_data.append(words)
        fin.close()

    def _data_processing(self):
        self._build_vocab()

    def _build_vocab(self):
        word_list = list()
        for text in self.text_data:
            word_list.extend(text)
        tokens = [token for token, _ in collections.Counter(word_list).items()]
        tokens = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token] + tokens
        tokens = tokens[:self.max_vocab_size]

        self.idx2token = dict(zip(range(self.max_vocab_size), tokens))
        self.token2idx = dict(zip(tokens, range(self.max_vocab_size)))

    def __len__(self):
        return len(self.text_data)

    def shuffle(self):
        # temp = list(zip(self.text_data, self.text_idx_data, self.idx_length_data))
        # random.shuffle(temp)
        # self.text_data[:], self.text_idx_data[:], self.idx_length_data[:] = zip(*temp)
        random.shuffle(self.text_data)

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
            tp_text_data = self.text_data[start: end]
            corpus_list.append(Corpus(idx2token=self.idx2token,
                                      token2idx=self.token2idx,
                                      target_text_data=tp_text_data)
                               )
        return corpus_list
