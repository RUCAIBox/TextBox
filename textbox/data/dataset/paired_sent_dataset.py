# @Time   : 2020/11/16
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


# UPDATE:
# @Time   : 2020/12/04
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

import os
import pickle
import nltk
import collections
import random
from textbox.data.dataset import Dataset


class PairedSentenceDataset(Dataset):
    def __init__(self, config):
        self.source_language = config['source_language'].lower()
        self.target_language = config['target_language'].lower()
        self.source_suffix = config['source_suffix']
        self.target_suffix = config['target_suffix']
        self.share_vocab = config['share_vocab']
        if config['target_max_vocab_size'] is None or config['source_max_vocab_size'] is None:
            self.source_max_vocab_size = config['max_vocab_size']
            self.target_max_vocab_size = config['max_vocab_size']
        else:
            self.source_max_vocab_size = config['source_max_vocab_size']
            self.target_max_vocab_size = config['target_max_vocab_size']

        if config['target_max_seq_length'] is None or config['source_max_seq_length'] is None:
            self.source_max_vocab_size = config['max_seq_length']
            self.target_max_vocab_size = config['max_seq_length']
        else:
            self.source_max_seq_length = config['source_max_seq_length']
            self.target_max_seq_length = config['target_max_seq_length']
        super().__init__(config)

    def _get_preset(self):
        self.source_token2idx = {}
        self.source_idx2token = {}
        self.target_token2idx = {}
        self.target_idx2token = {}
        self.source_text_data = []
        self.target_text_data = []

    def __len__(self):
        return sum([len(data) for data in self.source_text_data])

    def _load_data(self, dataset_path):
        """Load dataset from split (train, dev, test).
        This is designed for paired sentence format, such as translation task and summarization task.
        Args:
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
                words = nltk.word_tokenize(line.strip(), language=self.source_language)[:self.source_max_seq_length]
                source_text.append(words)
            fin.close()
            self.source_text_data.append(source_text)

            target_file = os.path.join(dataset_path, '{}.{}'.format(prefix, self.target_suffix))
            target_text = []
            fin = open(target_file, "r")
            for line in fin:
                words = nltk.word_tokenize(line.strip(), language=self.target_language)[:self.target_max_seq_length]
                target_text.append(words)
            fin.close()
            self.target_text_data.append(target_text)

    def _data_processing(self):
        self._build_vocab()

    def _build_vocab_text(self, text_data_list, max_vocab_size):
        word_list = list()
        for text_data in text_data_list:
            for text in text_data:
                word_list.extend(text)
        token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
        token_count.sort(reverse=True)
        tokens = [word for count, word in token_count]
        tokens = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token] + tokens
        tokens = tokens[:max_vocab_size]
        idx2token = dict(zip(range(max_vocab_size), tokens))
        token2idx = dict(zip(tokens, range(max_vocab_size)))
        return idx2token, token2idx

    def _build_vocab(self):
        if self.share_vocab:
            assert self.source_language == self.target_language
            text_data = self.source_text_data + self.target_text_data
            self.source_idx2token, self.source_token2idx = self._build_vocab_text(text_data,
                                                                                  self.source_max_vocab_size)
            self.target_idx2token, self.target_token2idx = self.source_idx2token, self.source_token2idx
            print("Share Vocabulary between source and target, vocab size: {}".format(len(self.target_idx2token)))
        else:
            self.source_idx2token, self.source_token2idx = self._build_vocab_text(self.source_text_data,
                                                                                  max_vocab_size=self.source_max_vocab_size)
            self.target_idx2token, self.target_token2idx = self._build_vocab_text(self.target_text_data,
                                                                                  max_vocab_size=self.target_max_vocab_size)
            print("Source vocab size: {}, Target vocab size: {}".format(len(self.source_idx2token),
                                                                        len(self.target_idx2token)))

    def shuffle(self):
        pass

    def detect_restored(self, dataset_path):
        required_files = []
        for prefix in ['train', 'dev', 'test']:
            for suffix in [self.source_suffix, self.target_suffix]:
                filename = os.path.join(dataset_path, '{}.{}.bin'.format(prefix, suffix))
                required_files.append(filename)
        src_vocab_file = os.path.join(dataset_path, '{}.vocab'.format(self.source_suffix))
        tar_vocab_file = os.path.join(dataset_path, '{}.vocab'.format(self.target_suffix))
        required_files.append(src_vocab_file)
        required_files.append(tar_vocab_file)
        absent_file_flag = False
        for filename in required_files:
            if not self.check_file_exist(filename):
                self.logger.info('File {} not exist'.format(filename))
                absent_file_flag = True
        if absent_file_flag:
            return False
        return True

    def _dump_data(self, dataset_path):
        info_str = ''
        src_vocab_file = os.path.join(dataset_path, '{}.vocab'.format(self.source_suffix))
        tar_vocab_file = os.path.join(dataset_path, '{}.vocab'.format(self.target_suffix))
        with open(src_vocab_file, "wb") as f_src_vocab:
            pickle.dump([self.source_token2idx, self.source_idx2token], f_src_vocab)
        with open(tar_vocab_file, "wb") as f_tar_vocab:
            pickle.dump([self.target_token2idx, self.target_idx2token], f_tar_vocab)
        self.logger.info("Vocab size: source {}, target {}".format(len(self.source_token2idx),
                                                                   len(self.target_token2idx)))
        for i, prefix in enumerate(['train', 'dev', 'test']):
            source_text_data = self.source_text_data[i]
            target_text_data = self.target_text_data[i]
            source_text_data = self._text2id(source_text_data, self.source_token2idx)
            target_text_data = self._text2id(target_text_data, self.target_token2idx)
            src_idx_filename = os.path.join(dataset_path, '{}.{}.bin'.format(prefix, self.source_suffix))
            tar_idx_filename = os.path.join(dataset_path, '{}.{}.bin'.format(prefix, self.target_suffix))
            with open(src_idx_filename, "wb") as f_src:
                pickle.dump(source_text_data, f_src)
            with open(tar_idx_filename, "wb") as f_tar:
                pickle.dump(target_text_data, f_tar)
            if prefix == 'test':
                info_str += '{}: {} cases'.format(prefix, len(source_text_data))
            else:
                info_str += '{}: {} cases, '.format(prefix, len(source_text_data))
        self.logger.info(info_str)
        self.logger.info("Dump finished!")

    def load_restored(self, dataset_path):
        """Load dataset from restored binary files (train, dev, test).
        Args:
            dataset_path (str): path of dataset dir.
        """
        src_vocab_file = os.path.join(dataset_path, '{}.vocab'.format(self.source_suffix))
        tar_vocab_file = os.path.join(dataset_path, '{}.vocab'.format(self.target_suffix))
        with open(src_vocab_file, "rb") as f_src:
            self.source_token2idx, self.source_idx2token = pickle.load(f_src)
        with open(tar_vocab_file, "rb") as f_tar:
            self.target_token2idx, self.target_idx2token = pickle.load(f_tar)
        self.logger.info("Restore Vocab!")
        for prefix in ['train', 'dev', 'test']:
            src_idx_filename = os.path.join(dataset_path, '{}.{}.bin'.format(prefix, self.source_suffix))
            tar_idx_filename = os.path.join(dataset_path, '{}.{}.bin'.format(prefix, self.target_suffix))
            with open(src_idx_filename, "rb") as f_src:
                source_text_data = pickle.load(f_src)
                source_text_data = self._id2text(source_text_data, self.source_idx2token)
            with open(tar_idx_filename, "rb") as f_tar:
                target_text_data = pickle.load(f_tar)
                target_text_data = self._id2text(target_text_data, self.target_idx2token)
            self.source_text_data.append(source_text_data)
            self.target_text_data.append(target_text_data)
        self.logger.info("Restore finished!")

    def build(self, eval_setting=None):
        info_str = ''
        corpus_list = []
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
