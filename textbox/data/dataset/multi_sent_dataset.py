# @Time   : 2021/2/3
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.multi_sent_dataset
########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import tokenize, split_data, build_vocab, detect_restored, dump_data, load_restored


class MultipleSentenceDataset(AbstractDataset):

    def __init__(self, config):
        self.language = config['language'].lower()
        self.max_vocab_size = config['max_vocab_size']
        self.group_split_token = config['group_split_token']
        self.sentence_split_token = config['sentence_split_token']
        self.combine_knw_src = config['combine_knw_src']

        self._build_data_format(config)
        super().__init__(config)

    def _build_data_format(self, config):
        for group in ['knowledge', 'source', 'target']:
            format_name = group + '_format'
            format = config[format_name] if format_name in config else 'none'
            setattr(self, format_name, format)

            if format != 'none':
                max_length_name = 'max_' + group + '_length'
                if max_length_name in config:
                    setattr(self, max_length_name, config[max_length_name])
                else:
                    setattr(self, max_length_name, config['max_seq_length'])

            if format == 'multiple':
                max_num_name = 'max_' + group + '_num'
                if max_num_name in config:
                    setattr(self, max_num_name, config[max_num_name])
                else:
                    setattr(self, max_num_name, config['max_sentence_num'])

    def _get_preset(self):
        self.token2idx = {}
        self.idx2token = {}
        self.group_text_data = [[], [], []]

    def _load_multi_data(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise ValueError('File {} not exist'.format(dataset_path))

        fin = open(dataset_path, "r")
        group_text = [[], [], []]
        for line in fin:
            groups = line.strip().lower().split(self.group_split_token)

            drop_flag = False
            for i, (group, data) in enumerate(zip(['target', 'source', 'knowledge'], groups[::-1])):
                max_length = getattr(self, 'max_' + group + '_length')
                if getattr(self, group + '_format') == 'single':
                    text = tokenize(data, self.tokenize_strategy, self.language)
                    drop_flag |= (len(text) > max_length)
                    text = text[:max_length]
                    group_text[i].append(text)
                else:
                    max_num = getattr(self, 'max_' + group + '_num')
                    texts = [
                        tokenize(text, self.tokenize_strategy, self.language)
                        for text in data.split(self.sentence_split_token)
                    ]
                    drop_flag |= any([len(text) > max_length for text in texts])
                    drop_flag |= (len(texts) > max_num)
                    texts = [text[:max_length] for text in texts[-max_num:]]
                    group_text[i].append(texts)

            if drop_flag & (self.overlength_strategy == 'drop'):
                group_text = [group[:-1] for group in group_text]
        return group_text[::-1]

    def _load_split_data(self, dataset_path):
        """Load dataset from split (train, dev, test).
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        for i, prefix in enumerate(['train', 'dev', 'test']):
            filename = os.path.join(dataset_path, '{}.txt'.format(prefix))
            knowledge, src, tgt = self._load_multi_data(filename)
            self.group_text_data[0].append(knowledge)
            self.group_text_data[1].append(src)
            self.group_text_data[2].append(tgt)

    def _load_single_data(self, dataset_path):
        """Load full corpus.
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        dataset_file = os.path.join(dataset_path, 'corpus.txt')
        group_text_data = self._load_multi_data(dataset_file)
        self.group_text_data = split_data([text_data for text_data in group_text_data], self.split_ratio)

    def _load_data(self, dataset_path):
        if self.split_strategy == "load_split":
            self._load_split_data(dataset_path)
        elif self.split_strategy == "by_ratio":
            self._load_single_data(dataset_path)
        else:
            raise NotImplementedError("{} split strategy not implemented".format(self.split_strategy))

        if self.combine_knw_src:
            self.knowledge_format = 'none'
            for i in range(3):
                self.group_text_data[1][i] = [
                    k + s for k, s in zip(self.group_text_data[0][i], self.group_text_data[1][i])
                ]

        for i, group in enumerate(['knowledge', 'source', 'target']):
            if getattr(self, group + '_format') != 'none':
                setattr(self, group + '_text_data', self.group_text_data[i])

    def _build_vocab(self):
        text_data = self.group_text_data[0] + self.group_text_data[1] + self.group_text_data[2]
        self.idx2token, self.token2idx, self.max_vocab_size = build_vocab(
            text_data, self.max_vocab_size, self.special_token_list
        )

    def _detect_restored(self, dataset_path):
        restored_flag = True
        for group in ['knowledge', 'source', 'target']:
            if getattr(self, group + '_format') != 'none':
                restored_flag &= detect_restored(dataset_path, group + '.', ignore_file='vocab')
        return restored_flag & detect_restored(dataset_path, ignore_file='data')

    def _dump_data(self, dataset_path):
        for group in ['knowledge', 'source', 'target']:
            if getattr(self, group + '_format') != 'none':
                dump_data(dataset_path, getattr(self, group + '_text_data'), suffix=group + '.')
        dump_data(dataset_path, idx2token=self.idx2token, token2idx=self.token2idx)
        self.logger.info("Dump finished!")

    def _load_restored(self, dataset_path):
        """Load dataset from restored binary files (train, dev, test).
        Args:
            dataset_path (str): path of dataset dir.
        """
        for group in ['knowledge', 'source', 'target']:
            if getattr(self, group + '_format') != 'none':
                text_data = load_restored(dataset_path, group + '.', ignore_file='vocab')[0]
                setattr(self, group + '_text_data', text_data)
        idx2token, token2idx = load_restored(dataset_path, ignore_file='data')
        setattr(self, 'idx2token', idx2token)
        setattr(self, 'token2idx', token2idx)
        self.max_vocab_size = len(self.idx2token)
        self.logger.info("Restore finished!")

    def build(self):
        info_str = ''
        corpus_list = []
        self.logger.info("Vocab size: {}".format(self.max_vocab_size))

        for i, prefix in enumerate(['train', 'dev', 'test']):
            tp_data = {
                'idx2token': self.idx2token,
                'token2idx': self.token2idx,
                'vocab_size': self.max_vocab_size,
                'max_source_length': self.max_source_length,
                'max_target_length': self.max_target_length,
            }
            for group in ['knowledge', 'source', 'target']:
                if getattr(self, group + '_format') != 'none':
                    text_data = getattr(self, group + '_text_data')[i]
                    tp_data[group + '_text_data'] = text_data
            corpus_list.append(tp_data)
            info_str += '{}: {} cases, '.format(prefix, len(tp_data['target_text_data']))

        self.logger.info(info_str[:-2] + '\n')
        return corpus_list
