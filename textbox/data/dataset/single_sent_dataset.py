# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

import os
import nltk
import collections
import random
from textbox.data.dataset import Dataset


class SingleSentenceDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        self.source_language = config['source_language'].lower()
        self.strategy = config['split_strategy']
        assert self.strategy is not None
        self.split_ratio = config['split_ratio']
        super().__init__(config, saved_dataset)

    def _get_preset(self):
        self.token2idx = {}
        self.idx2token = {}
        self.text_data = []

    def _load_splitted_data(self, dataset_path):
        """Load dataset from split (train, dev, test).
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        train_src_file = os.path.join(dataset_path, 'train.txt')
        if not os.path.isfile(train_src_file):
            raise ValueError('File {} not exist'.format(train_src_file))
        for prefix in ['train', 'dev', 'test']:
            source_file = os.path.join(dataset_path, '{}.txt'.format(prefix))
            source_text = []
            fin = open(source_file, "r")
            for line in fin:
                words = nltk.word_tokenize(line.strip(), language=self.source_language)[:self.max_seq_length]
                source_text.append(words)
            fin.close()
            self.text_data.append(source_text)

    def _load_single_data(self, dataset_path):
        """Load full corpus.
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        dataset_file = os.path.join(dataset_path, 'corpus_large.txt')
        if not os.path.isfile(dataset_file):
            raise ValueError('File {} not exist'.format(dataset_file))

        fin = open(dataset_file, "r")
        for line in fin:
            words = nltk.word_tokenize(line.strip(), language=self.source_language.lower())[:self.max_seq_length]
            self.text_data.append(words)
        fin.close()

    def _load_data(self, dataset_path):
        if self.strategy == "load_split":
            self._load_splitted_data(dataset_path)
        elif self.strategy == "by_ratio":
            self._load_single_data(dataset_path)
        else:
            raise NotImplementedError("{} split strategy not implemented".format(self.strategy))

    def _data_processing(self):
        self._build_vocab()

    def _build_vocab(self):
        if self.strategy == "load_split":
            word_list = list()
            for sent_list in self.text_data:
                for text in sent_list:
                    word_list.extend(text)
            token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
            token_count.sort(reverse=True)
            tokens = [word for count, word in token_count]
            tokens = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token] + tokens
            tokens = tokens[:self.max_vocab_size]

            self.idx2token = dict(zip(range(self.max_vocab_size), tokens))
            self.token2idx = dict(zip(tokens, range(self.max_vocab_size)))
        elif self.strategy == "by_ratio":
            word_list = list()
            for text in self.text_data:
                word_list.extend(text)
            token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
            token_count.sort(reverse=True)
            tokens = [word for count, word in token_count]
            tokens = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token] + tokens
            tokens = tokens[:self.max_vocab_size]

            self.idx2token = dict(zip(range(self.max_vocab_size), tokens))
            self.token2idx = dict(zip(tokens, range(self.max_vocab_size)))
        else:
            raise NotImplementedError("{} split strategy not implemented".format(self.strategy))

    def __len__(self):
        return len(self.text_data)

    def shuffle(self):
        # temp = list(zip(self.text_data, self.text_idx_data, self.idx_length_data))
        # random.shuffle(temp)
        # self.text_data[:], self.text_idx_data[:], self.idx_length_data[:] = zip(*temp)
        random.shuffle(self.text_data)

    def split_by_ratio(self, ratios):
        self.logger.debug('split by ratios [{}]'.format(ratios))
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        tot_cnt = self.__len__()
        split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
        corpus_list = []
        for start, end in zip([0] + split_ids, split_ids + [tot_cnt]):
            tp_text_data = self.text_data[start: end]
            tp_data = {
                'idx2token': self.idx2token,
                'token2idx': self.token2idx,
                'text_data': tp_text_data
            }
            corpus_list.append(tp_data)
        return corpus_list

    def load_split(self):
        """Load splitted dataset.
        """
        corpus_list = []
        for sent_list in self.text_data:
            tp_data = {
                'idx2token': self.idx2token,
                'token2idx': self.token2idx,
                'text_data': sent_list
            }
            corpus_list.append(tp_data)
        return corpus_list

    def build(self):
        self.shuffle()

        split_args = {'strategy': self.strategy, 'ratios': self.split_ratio}
        if split_args['strategy'] == 'by_ratio':
            corpus_list = self.split_by_ratio(split_args['ratios'])
        elif split_args['strategy'] == 'load_split':
            corpus_list = self.load_split()
        else:
            raise NotImplementedError()
        return corpus_list
