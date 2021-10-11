# @Time   : 2021/1/30
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/10
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.attr_sent_dataset
########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import build_vocab, text2idx


class AttributedSentenceDataset(AbstractDataset):

    def __init__(self, config):
        super().__init__(config)

    def _get_preset(self):
        super()._get_preset()
        self.source_text = []
        self.target_text = []

    def _load_attribute(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise ValueError('File {} not exist'.format(dataset_path))

        attribute_data = []
        with open(dataset_path, "r") as fin:
            for line in fin:
                attribute = line.strip().split('\t')
                attribute_data.append(attribute)
        return attribute_data

    def _load_source_data(self):
        for i, prefix in enumerate(['train', 'valid', 'test']):
            filename = os.path.join(self.dataset_path, f'{prefix}.src')

            text_data = self._load_attribute(filename)
            assert len(text_data) == len(self.target_text[i])
            self.source_text.append(text_data)

    def _build_attribute_vocab(self):
        attribute_num = len(self.source_text[0][0])
        attribute_set = [set() for _ in range(attribute_num)]
        for group in self.source_text:
            for attribute in group:
                assert len(attribute) == attribute_num
                for i, attr in enumerate(attribute):
                    attribute_set[i].add(attr)

        self.source_idx2token = []
        self.source_token2idx = []
        for i in range(attribute_num):
            attribute = list(attribute_set[i])
            attribute_size = len(attribute)
            self.source_idx2token.append(dict(zip(range(attribute_size), attribute)))
            self.source_token2idx.append(dict(zip(attribute, range(attribute_size))))

    def _build_vocab(self):
        self.target_idx2token, self.target_token2idx, self.target_vocab_size = build_vocab(
            self.target_text, self.target_vocab_size, self.special_token_list
        )
        self._build_attribute_vocab()

    def _attribute2idx(self, source_text, source_token2idx):
        new_idx = []
        for group in source_text:
            idx = []
            for sent in group:
                sent_idx = [source_token2idx[i][attr] for i, attr in enumerate(sent)]
                idx.append(sent_idx)
            new_idx.append(idx)
        return new_idx

    def _text2idx(self):
        self.source_idx = self._attribute2idx(self.source_text, self.source_token2idx)
        self.target_idx, self.target_length, _ = text2idx(
            self.target_text, self.target_token2idx, self.tokenize_strategy
        )
