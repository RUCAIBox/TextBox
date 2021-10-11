# @Time   : 2020/11/16
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/10, 2021/1/29, 2020/12/04
# @Author : Tianyi Tang, Gaole He
# @Email  : steven_tang@ruc.edu.cn, hegaole@ruc.edu.cn

"""
textbox.data.dataset.paired_sent_dataset
########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import load_data, build_vocab, text2idx


class PairedSentenceDataset(AbstractDataset):

    def __init__(self, config):
        self.share_vocab = config['share_vocab']
        super().__init__(config)

    def _get_preset(self):
        super()._get_preset()
        self.source_text = []
        self.target_text = []

    def _load_source_data(self):
        for i, prefix in enumerate(['train', 'valid', 'test']):
            filename = os.path.join(self.dataset_path, f'{prefix}.src')
            text_data = load_data(
                filename, self.tokenize_strategy, self.source_max_length, self.source_language,
                self.source_multi_sentence, self.source_max_num
            )
            assert len(text_data) == len(self.target_text[i])
            self.source_text.append(text_data)

    def _build_vocab(self):
        if self.share_vocab:
            assert self.source_vocab_size == self.target_vocab_size
            text_data = self.source_text + self.target_text
            self.source_idx2token, self.source_token2idx, self.source_vocab_size = build_vocab(
                text_data, self.source_vocab_size, self.special_token_list
            )
            self.target_idx2token, self.target_token2idx, self.target_vocab_size = self.source_idx2token, self.source_token2idx, self.source_vocab_size
        else:
            self.source_idx2token, self.source_token2idx, self.source_vocab_size = build_vocab(
                self.source_text, self.source_vocab_size, self.special_token_list
            )
            self.target_idx2token, self.target_token2idx, self.target_vocab_size = build_vocab(
                self.target_text, self.target_vocab_size, self.special_token_list
            )

    def _text2idx(self):
        self.source_idx, self.source_length, self.source_num = text2idx(
            self.source_text, self.source_token2idx, self.tokenize_strategy
        )
        self.target_idx, self.target_length, self.target_num = text2idx(
            self.target_text, self.target_token2idx, self.tokenize_strategy
        )
