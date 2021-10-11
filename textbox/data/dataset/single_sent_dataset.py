# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/10, 2021/1/29, 2021/10/9
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.single_sent_dataset
########################################
"""

from textbox.data.dataset import AbstractDataset
from textbox.data.utils import build_vocab, text2idx


class SingleSentenceDataset(AbstractDataset):

    def __init__(self, config):
        super().__init__(config)

    def _get_preset(self):
        super()._get_preset()
        self.target_text = []

    def _load_source_data(self):
        pass

    def _build_vocab(self):
        self.target_idx2token, self.target_token2idx, self.target_vocab_size = build_vocab(
            self.target_text, self.target_vocab_size, self.special_token_list
        )

    def _text2idx(self):
        self.target_idx, self.target_length, _ = text2idx(
            self.target_text, self.target_token2idx, self.tokenize_strategy
        )
