# @Time   : 2021/10/12
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.wikibio_sent_dataset
##########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import load_data, build_vocab, text2idx


class WikiBioSentenceDataset(AbstractDataset):

    def __init__(self, config):
        super().__init__(config)

    def _get_preset(self):
        super()._get_preset()
        self.source_value_text = []
        self.source_key_text = []
        self.target_text = []

    def _load_source_data(self):
        for i, prefix in enumerate(['train', 'valid', 'test']):
            filename = os.path.join(self.dataset_path, f'{prefix}.src')

            text_data = load_data(
                filename, self.tokenize_strategy, self.source_max_length, self.source_language,
                True, self.source_max_num
            )
            assert len(text_data) == len(self.target_text[i])
            key_data = []
            for doc in text_data:
                key = []
                for kv in doc:
                    k, kv[0] = kv[0].split('<kv>')
                    key.append(k)
                key_data.append(key)
            self.source_value_text.append(text_data)
            self.source_key_text.append(key_data)

    def _build_vocab(self):
        self.source_key_idx2token, self.source_key_token2idx, _ = build_vocab(
            self.source_key_text, self.source_vocab_size, self.special_token_list
        )
        data = self.source_value_text + self.target_text
        self.source_idx2token, self.source_token2idx, self.source_vocab_size = build_vocab(
            data, self.source_vocab_size, self.special_token_list
        )
        self.target_idx2token, self.target_token2idx, self.target_vocab_size = self.source_idx2token, self.source_token2idx, self.source_vocab_size

    def _text2idx(self):
        self.source_key_idx, _, _ = text2idx(
            self.source_key_text, self.source_key_token2idx, 'none'
        )
        self.source_value_idx, _, _ = text2idx(
            self.source_value_text, self.source_token2idx, 'none'
        )
        self.target_idx, self.target_length, _ = text2idx(
            self.target_text, self.target_token2idx, self.tokenize_strategy
        )
