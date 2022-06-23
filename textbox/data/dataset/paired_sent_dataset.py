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


class CopyPairedSentenceDataset(PairedSentenceDataset):
    def __init__(self, config):
        super(CopyPairedSentenceDataset, self).__init__(config)

    def _text2idx(self):
        data_dict = self.text2idx(self.source_text, self.target_text, self.target_token2idx, self.sos_token_idx,
                                  self.eos_token_idx, self.unknown_token_idx, self.config['is_pgen'])

        for key, value in data_dict.items():
            setattr(self, key, value)

    @staticmethod
    def text2idx(source_text, target_text, token2idx, sos_idx, eos_idx, unk_idx, is_pgen=False):
        data_dict = {'source_idx': [], 'source_length': [],
                     'target_input_idx': [], 'target_output_idx': [], 'target_length': []}

        if is_pgen:
            data_dict['source_extended_idx'] = []
            data_dict['source_oovs'] = []

        def article2ids(article_words):
            ids = []
            oovs = []
            for w in article_words:
                i = token2idx.get(w, unk_idx)
                if i == unk_idx:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(token2idx) + oov_num)
                else:
                    ids.append(i)
            return ids, oovs

        def abstract2ids(abstract_words, article_oovs):
            ids = []
            for w in abstract_words:
                i = token2idx.get(w, unk_idx)
                if i == unk_idx:
                    if w in article_oovs:
                        vocab_idx = len(token2idx) + article_oovs.index(w)
                        ids.append(vocab_idx)
                    else:
                        ids.append(unk_idx)
                else:
                    ids.append(i)
            return ids

        for i, prefix in enumerate(['train', 'valid', 'test']):
            new_source_idx = []
            new_source_length = []
            new_target_input_idx = []
            new_target_output_idx = []
            new_target_length = []
            if is_pgen:
                new_source_extended_idx = []
                new_source_oovs = []
            for source_sent, target_sent in zip(source_text[i], target_text[i]):
                source_idx = [token2idx.get(word, unk_idx) for word in source_sent]
                target_input_idx = [sos_idx] + [token2idx.get(word, unk_idx) for word in target_sent]

                if is_pgen:
                    source_extended_idx, source_oovs = article2ids(source_sent)
                    target_output_idx = abstract2ids(target_sent, source_oovs) + [eos_idx]
                    new_source_extended_idx.append(source_extended_idx)
                    new_source_oovs.append(source_oovs)
                else:
                    target_output_idx = [token2idx.get(word, unk_idx) for word in target_sent] + [eos_idx]

                new_source_idx.append(source_idx)
                new_source_length.append(len(source_idx))
                new_target_input_idx.append(target_input_idx)
                new_target_output_idx.append(target_output_idx)
                new_target_length.append(len(target_input_idx))

            data_dict['source_idx'].append(new_source_idx)
            data_dict['source_length'].append(new_source_length)
            data_dict['target_input_idx'].append(new_target_input_idx)
            data_dict['target_output_idx'].append(new_target_output_idx)
            data_dict['target_length'].append(new_target_length)

            if is_pgen:
                data_dict['source_extended_idx'].append(new_source_extended_idx)
                data_dict['source_oovs'].append(new_source_oovs)

        return data_dict
