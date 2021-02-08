# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/1/29
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.single_sent_dataset
########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import load_data, split_data, build_vocab, detect_restored, dump_data, load_restored


class SingleSentenceDataset(AbstractDataset):

    def __init__(self, config):
        self.language = config['language'].lower()
        self.max_vocab_size = config['max_vocab_size']
        self.max_seq_length = config['max_seq_length']
        super().__init__(config)

    def __len__(self):
        return sum([len(data) for data in self.text_data])

    def _get_preset(self):
        self.idx2token = {}
        self.token2idx = {}
        self.text_data = []

    def _load_split_data(self, dataset_path):
        """Load dataset from split (train, dev, test).
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        for prefix in ['train', 'dev', 'test']:
            filename = os.path.join(dataset_path, '{}.txt'.format(prefix))
            text_data = load_data(
                filename, self.tokenize_strategy, self.overlength_strategy, self.max_seq_length, self.language
            )
            self.text_data.append(text_data)

    def _load_single_data(self, dataset_path):
        """Load full corpus.
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        dataset_file = os.path.join(dataset_path, 'corpus.txt')
        self.text_data = load_data(
            dataset_file, self.tokenize_strategy, self.overlength_strategy, self.max_seq_length, self.language
        )
        self.text_data = split_data([self.text_data], self.split_ratio)[0]

    def _load_data(self, dataset_path):
        if self.split_strategy == "load_split":
            self._load_split_data(dataset_path)
        elif self.split_strategy == "by_ratio":
            self._load_single_data(dataset_path)
        else:
            raise NotImplementedError("{} split strategy not implemented".format(self.split_strategy))

    def _build_vocab(self):
        self.idx2token, self.token2idx, self.max_vocab_size = build_vocab(
            self.text_data, self.max_vocab_size, self.special_token_list
        )

    def _detect_restored(self, dataset_path):
        return detect_restored(dataset_path)

    def _dump_data(self, dataset_path):
        dump_data(dataset_path, self.text_data, self.idx2token, self.token2idx)
        self.logger.info("Dump finished!")

    def _load_restored(self, dataset_path):
        """Load dataset from restored binary files (train, dev, test).
        Args:
            dataset_path (str): path of dataset dir.
        """
        self.text_data, self.idx2token, self.token2idx = load_restored(dataset_path)
        self.max_vocab_size = len(self.idx2token)
        self.logger.info("Restore finished!")

    def build(self):
        info_str = ''
        corpus_list = []
        self.logger.info("Vocab size: {}".format(self.max_vocab_size))

        for i, prefix in enumerate(['train', 'dev', 'test']):
            text_data = self.text_data[i]
            tp_data = {
                'idx2token': self.idx2token,
                'token2idx': self.token2idx,
                'text_data': text_data,
            }
            corpus_list.append(tp_data)
            info_str += '{}: {} cases, '.format(prefix, len(text_data))

        self.logger.info(info_str[:-2] + '\n')
        return corpus_list
