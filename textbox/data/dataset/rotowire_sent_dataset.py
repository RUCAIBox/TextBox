# @Time   : 2021/10/12
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.rotowire_sent_dataset
###########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import build_vocab, text2idx, build_attribute_vocab, attribute2idx


class RotoWireSentenceDataset(AbstractDataset):

    def __init__(self, config):
        super().__init__(config)

    def _get_preset(self):
        super()._get_preset()
        self.source_text = []
        self.source_plan_text = []
        self.target_text = []

    def _load_table_data(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise ValueError('File {} not exist'.format(dataset_path))
        
        table_data = []
        with open(dataset_path, "r") as fin:
            for line in fin:
                line = line.strip().split('\t')
                line = [[line[i*4], line[i*4+1], line[i*4+2], line[i*4+3]] for i in range(len(line)//4)]
                table_data.append(line)
        return table_data

    def _load_plan_data(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise ValueError('File {} not exist'.format(dataset_path))

        plan_data = []
        with open(dataset_path, "r") as fin:
            for line in fin:
                line = line.strip().split(' ')
                plan_data.append(line)
        return plan_data

    def _load_source_data(self):
        for i, prefix in enumerate(['train', 'valid', 'test']):
            filename = os.path.join(self.dataset_path, f'{prefix}.src')
            text_data = self._load_table_data(filename)
            assert len(text_data) == len(self.target_text[i])
            self.source_text.append(text_data)

        for i, prefix in enumerate(['train', 'valid']):
            filename = os.path.join(self.dataset_path, f'{prefix}.plan')
            text_data = self._load_plan_data(filename)
            assert len(text_data) == len(self.target_text[i])
            self.source_plan_text.append(text_data)

    def _build_vocab(self):
        self.source_idx2token, self.source_token2idx = build_attribute_vocab(self.source_text)
        self.target_idx2token, self.target_token2idx, self.target_vocab_size = build_vocab(
                self.target_text, self.target_vocab_size, self.special_token_list
            )

    def _text2idx(self):
        self.source_idx, self.source_length = attribute2idx(self.source_text, self.source_token2idx)
        self.source_plan_idx = [[[int(idx) for idx in doc] for doc in group] for group in self.source_plan_text]
        self.source_plan_length = [[len(doc) for doc in group] for group in self.source_plan_text]
        self.target_idx, self.target_length, _ = text2idx(
            self.target_text, self.target_token2idx, self.tokenize_strategy
        )
