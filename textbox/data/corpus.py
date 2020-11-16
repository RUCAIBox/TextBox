# @Time   : 2020/7/10
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time    : 2020/9/15, 2020/9/16, 2020/8/12
# @Author  : Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email   : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
textbox.data.corpus
############################
"""

from textbox.data.abstract_corpus import AbstractCorpus


class Corpus(AbstractCorpus):
    
    def __init__(self, idx2token, token2idx, target_text_data, source_text_data=None):
        super().__init__(idx2token, token2idx)
        self.source_text_data = source_text_data
        self.target_text_data = target_text_data
        self._build_source_data()
        self._build_target_data()

    def _build_source_data(self):
        if self.source_text_data is None:
            self.source_text_idx_data = None
            self.source_idx_length_data = None
        else:
            self.source_text_idx_data = []
            self.source_idx_length_data = []
            for text in self.source_text_data:
                text_idx = [self._token2idx(token) for token in text]
                self.source_text_idx_data.append(text_idx)
                self.source_idx_length_data.append(len(text_idx))

    def _build_target_data(self):
        self.target_text_idx_data = []
        self.target_idx_length_data = []
        for text in self.target_text_data:
            text_idx = [self.sos_token_idx] + [self._token2idx(token) for token in text] + [self.eos_token_idx]
            self.target_text_idx_data.append(text_idx)
            self.target_idx_length_data.append(len(text_idx))
