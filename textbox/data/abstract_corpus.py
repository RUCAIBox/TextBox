# @Time   : 2020/7/10
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time    : 2020/9/15, 2020/9/16, 2020/8/12
# @Author  : Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email   : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
textbox.data.abstract_corpus
############################
"""

from textbox.utils.enum_type import SpecialTokens


class AbstractCorpus(object):

    def __init__(self, idx2token, token2idx):
        self.idx2token = idx2token
        self.token2idx = token2idx

        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS

    def _idx2token(self, inputs):
        if isinstance(inputs, list):
            return [self._idx2token[x] for x in inputs]
        return self.idx2token[inputs]

    def _token2idx(self, inputs):
        if isinstance(inputs, list):
            return [self._token2idx[x] for x in inputs]
        return self.token2idx.get(inputs, self.unknown_token_idx)

    @property
    def padding_token_idx(self):
        r"""The `int` index of the special token indicating the padding token.
        """
        return self.token2idx[self.padding_token]

    @property
    def unknown_token_idx(self):
        r"""The `int` index of the special token indicating the unknown token.
        """
        return self.token2idx[self.unknown_token]

    @property
    def sos_token_idx(self):
        r"""The `int` index of the special token indicating the start of sequence.
        """
        return self.token2idx[self.sos_token]

    @property
    def eos_token_idx(self):
        r"""The `int` index of the special token indicating the end of sequence.
        """
        return self.token2idx[self.eos_token]