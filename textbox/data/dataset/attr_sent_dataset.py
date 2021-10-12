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
from textbox.data.utils import build_vocab, text2idx, build_attribute_vocab, attribute2idx


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

    def _build_vocab(self):
        self.source_idx2token, self.source_token2idx = build_attribute_vocab(self.source_text)
        self.target_idx2token, self.target_token2idx, self.target_vocab_size = build_vocab(
            self.target_text, self.target_vocab_size, self.special_token_list
        )

    def _text2idx(self):
        self.source_idx = attribute2idx(self.source_text, self.source_token2idx)
        self.target_idx, self.target_length, _ = text2idx(
            self.target_text, self.target_token2idx, self.tokenize_strategy
        )
