# @Time   : 2020/11/16
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/1/29, 2020/12/04
# @Author : Tianyi Tang, Gaole He
# @Email  : steven_tang@ruc.edu.cn, hegaole@ruc.edu.cn

"""
textbox.data.dataset.paired_sent_dataset
########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import load_data, split_data, build_vocab, detect_restored, dump_data, load_restored


class PairedSentenceDataset(AbstractDataset):

    def __init__(self, config):
        self.source_language = config['source_language'].lower()
        self.target_language = config['target_language'].lower()
        self.source_suffix = config['source_suffix'].lower()
        self.target_suffix = config['target_suffix'].lower()
        self.share_vocab = config['share_vocab']

        if config['target_max_vocab_size'] is None or config['source_max_vocab_size'] is None:
            self.source_max_vocab_size = config['max_vocab_size']
            self.target_max_vocab_size = config['max_vocab_size']
        else:
            self.source_max_vocab_size = config['source_max_vocab_size']
            self.target_max_vocab_size = config['target_max_vocab_size']

        if config['target_max_seq_length'] is None or config['source_max_seq_length'] is None:
            self.source_max_seq_length = config['max_seq_length']
            self.target_max_seq_length = config['max_seq_length']
        else:
            self.source_max_seq_length = config['source_max_seq_length']
            self.target_max_seq_length = config['target_max_seq_length']
        super().__init__(config)

    def __len__(self):
        return sum([len(data) for data in self.source_text_data])

    def _get_preset(self):
        self.source_token2idx = {}
        self.source_idx2token = {}
        self.target_token2idx = {}
        self.target_idx2token = {}
        self.source_text_data = []
        self.target_text_data = []

    def _load_paired_data(self, source_file, target_file):
        if self.overlength_strategy == 'drop':
            loaded_source_text = load_data(
                source_file, self.tokenize_strategy, 'none', self.source_max_seq_length, self.source_language
            )
            loaded_target_text = load_data(
                target_file, self.tokenize_strategy, 'none', self.target_max_seq_length, self.target_language
            )
            assert len(loaded_source_text) == len(loaded_target_text)
            source_text = []
            target_text = []
            for src, tgt in zip(loaded_source_text, loaded_target_text):
                if (len(src) <= self.source_max_seq_length and len(tgt) <= self.target_max_seq_length):
                    source_text.append(src)
                    target_text.append(tgt)
        else:
            source_text = load_data(
                source_file, self.tokenize_strategy, self.overlength_strategy, self.source_max_seq_length,
                self.source_language
            )
            target_text = load_data(
                target_file, self.tokenize_strategy, self.overlength_strategy, self.target_max_seq_length,
                self.target_language
            )

        return source_text, target_text

    def _load_split_data(self, dataset_path):
        """Load dataset from split (train, dev, test).
        This is designed for paired sentence format, such as translation task and summarization task.
        
        Args:
            dataset_path (str): path of dataset dir.
        """
        for prefix in ['train', 'dev', 'test']:
            source_file = os.path.join(dataset_path, '{}.{}'.format(prefix, self.source_suffix))
            target_file = os.path.join(dataset_path, '{}.{}'.format(prefix, self.target_suffix))

            source_text, target_text = self._load_paired_data(source_file, target_file)

            self.source_text_data.append(source_text)
            self.target_text_data.append(target_text)

    def _load_single_data(self, dataset_path):
        """Load full corpus.
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        source_file = os.path.join(dataset_path, 'source.txt')
        target_file = os.path.join(dataset_path, 'target.txt')

        source_text, target_text = self._load_paired_data(source_file, target_file)

        self.source_text_data, self.target_text_data = split_data([source_text, target_text], self.split_ratio)

    def _load_data(self, dataset_path):
        if self.split_strategy == "load_split":
            self._load_split_data(dataset_path)
        elif self.split_strategy == "by_ratio":
            self._load_single_data(dataset_path)
        else:
            raise NotImplementedError("{} split strategy not implemented".format(self.split_strategy))

    def _build_vocab(self):
        if self.share_vocab:
            assert self.source_language == self.target_language
            text_data = self.source_text_data + self.target_text_data
            self.source_idx2token, self.source_token2idx, self.source_max_vocab_size = build_vocab(
                text_data, self.source_max_vocab_size, self.special_token_list
            )
            self.target_idx2token, self.target_token2idx = self.source_idx2token, self.source_token2idx
        else:
            self.source_idx2token, self.source_token2idx, self.source_max_vocab_size = build_vocab(
                self.source_text_data, self.source_max_vocab_size, self.special_token_list
            )
            self.target_idx2token, self.target_token2idx, self.target_max_vocab_size = build_vocab(
                self.target_text_data, self.target_max_vocab_size, self.special_token_list
            )

    def _detect_restored(self, dataset_path):
        return detect_restored(dataset_path,
                               self.source_suffix + '.') and detect_restored(dataset_path, self.target_suffix + '.')

    def _dump_data(self, dataset_path):
        dump_data(
            dataset_path, self.source_text_data, self.source_idx2token, self.source_token2idx, self.source_suffix + '.'
        )
        dump_data(
            dataset_path, self.target_text_data, self.target_idx2token, self.target_token2idx, self.target_suffix + '.'
        )
        self.logger.info("Dump finished!")

    def _load_restored(self, dataset_path):
        """Load dataset from restored binary files (train, dev, test).

        Args:
            dataset_path (str): path of dataset dir.
        """
        self.source_text_data, self.source_idx2token, self.source_token2idx = load_restored(
            dataset_path, self.source_suffix + '.'
        )
        self.target_text_data, self.target_idx2token, self.target_token2idx = load_restored(
            dataset_path, self.target_suffix + '.'
        )
        self.source_max_vocab_size = len(self.source_idx2token)
        self.target_max_vocab_size = len(self.target_idx2token)
        self.logger.info("Restore finished!")

    def build(self, is_print_log=True):
        info_str = ''
        corpus_list = []
        if (is_print_log):
            self.logger.info(
                "Vocab size: source {}, target {}".format(self.source_max_vocab_size, self.target_max_vocab_size)
            )

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
            info_str += '{}: {} cases, '.format(prefix, len(source_text_data))
        if (is_print_log):
            self.logger.info(info_str[:-2] + '\n')
        return corpus_list
