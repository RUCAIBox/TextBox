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


class TranslationDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset)
        self.source_language = config['source_language']
        self.target_language = config['target_language']
        self.source_suffix = config['source_suffix']
        self.target_suffix = config['target_suffix']
        try:
            self.source_tokenizer = nltk.data.load('tokenizers/punkt/{}.pickle'.format(self.source_language.lower()))
        except FileNotFoundError:
            print("Error occur when fetching tokenizers/punkt/{}.pickle".format(self.source_language.lower()))
        try:
            self.target_tokenizer = nltk.data.load('tokenizers/punkt/{}.pickle'.format(self.target_language.lower()))
        except FileNotFoundError:
            print("Error occur when fetching tokenizers/punkt/{}.pickle".format(self.target_language.lower()))

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        self.source_token2idx = {}
        self.source_idx2token = {}
        self.target_token2idx = {}
        self.target_idx2token = {}
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
        train_src_file = os.path.join(dataset_path, 'train.' + self.source_suffix)
        if not os.path.isfile(train_src_file):
            raise ValueError('File {} not exist'.format(train_src_file))
        for prefix in ['train', 'dev', 'test']:
            source_file = os.path.join(dataset_path, '{}.{}'.format(prefix, self.source_suffix))
            source_text = []
            fin = open(source_file, "r")
            for line in fin:
                words = self.source_tokenizer(line.strip())[:self.max_seq_length]
                # words = nltk.word_tokenize(line.strip())[:self.max_seq_length]
                source_text.append(words)
            fin.close()
            self.source_text_data.append(source_text)

            target_file = os.path.join(dataset_path, '{}.{}'.format(prefix, self.target_suffix))
            target_text = []
            fin = open(target_file, "r")
            for line in fin:
                words = self.target_tokenizer(line.strip())[:self.max_seq_length]
                # words = nltk.word_tokenize(line.strip())[:self.max_seq_length]
                target_text.append(words)
            fin.close()
            self.target_text_data.append(target_text)

    def _data_processing(self):
        self._build_vocab()

    def _build_vocab_text(self, text_data_list):
        word_list = list()
        for text_data in text_data_list:
            for text in text_data:
                word_list.extend(text)
        tokens = [token for token, _ in collections.Counter(word_list).items()]
        tokens = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token] + tokens
        tokens = tokens[:self.max_vocab_size]
        idx2token = dict(zip(range(self.max_vocab_size), tokens))
        token2idx = dict(zip(tokens, range(self.max_vocab_size)))
        return idx2token, token2idx

    def _build_vocab(self):
        self.source_idx2token, self.source_token2idx = self._build_vocab_text(self.source_text_data)
        self.target_idx2token, self.target_token2idx = self._build_vocab_text(self.target_text_data)

    def shuffle(self):
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
        info_str = ''
        for i, prefix in enumerate(['train', 'dev', 'test']):
            source_text_data = self.source_text_data[i]
            target_text_data = self.target_text_data[i]
            tp_data = {
                'source_idx2token': self.source_idx2token,
                'source_token2idx': self.source_token2idx,
                'source_text_data': source_text_data,
                'target_idx2token': self.target_idx2token,
                'target_token2idx': self.target_token2idx,
                'target_text_data': target_text_data
            }
            corpus_list.append(tp_data)
            if prefix == 'test':
                info_str += '{}: {} cases'.format(prefix, len(source_text_data))
            else:
                info_str += '{}: {} cases, '.format(prefix, len(source_text_data))
        self.logger.info(info_str)
        return corpus_list
