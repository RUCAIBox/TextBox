# @Time   : 2021/2/3
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.multi_sent_dataloader
################################################
"""

import random
import math
import torch

from textbox.data.dataloader.abstract_dataloader import AbstractDataLoader


class MultipleSentenceDataLoader(AbstractDataLoader):
    r""":class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (SingleSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, shuffle=shuffle)
        self.data_preprocess(dataset)

    def _build_multi_data(self, text_data, token2idx, need_text_start_end=True):
        r"""transform text to id and add sos and eos token index.
        input:
            text_data: list -> list -> list -> character, original text
            token2idx: dict, map token to index
            need_text_start_end, bool, indicates whether we should add sos and eos token index.
        output:
            text_idx_data: list -> list -> list -> int, list of word index
            text_idx_length_data: list -> list -> int, list of sequence length
            sent_num_data: list of sentence number
        """
        text_idx_data = []
        text_idx_length_data = []
        sent_num_data = []
        for sens in text_data:
            sens_idx_data = []
            idx_length = []
            for sen in sens:
                text_idx = self._token2idx(sen, token2idx)
                if need_text_start_end:
                    text_idx = [self.sos_token_idx] + text_idx + [self.eos_token_idx]
                sens_idx_data.append(text_idx)
                idx_length.append(len(text_idx))
            text_idx_data.append(sens_idx_data)
            text_idx_length_data.append(idx_length)
            sent_num_data.append(len(sens_idx_data))
        return text_idx_data, text_idx_length_data, sent_num_data

    def data_preprocess(self, dataset):
        required_key_list = [
            'idx2token', 'token2idx', 'knowledge_text_data', 'source_text_data', 'target_text_data'
        ]
        for dataset_attr in required_key_list:
            assert dataset_attr in dataset
            setattr(self, dataset_attr, dataset[dataset_attr])
        self.knowledge_text_idx_data, self.knowledge_idx_length_data, self.knowledge_sent_num_data = self._build_multi_data(
            self.knowledge_text_data, self.token2idx
        )
        self.source_text_idx_data, self.source_idx_length_data, self.source_sent_num_data = self._build_multi_data(
            self.source_text_data, self.token2idx
        )
        self.target_text_idx_data, self.target_idx_length_data = self._build_data(
            self.target_text_data, self.token2idx
        )

    def get_reference(self):
        return self.target_text_data

    @property
    def pr_end(self):
        return len(self.target_text_idx_data)

    def __len__(self):
        return math.ceil(len(self.target_text_idx_data) / self.batch_size)

    def _shuffle(self):
        temp = list(
            zip(
                self.knowledge_text_idx_data, self.knowledge_idx_length_data, self.knowledge_sent_num_data,
                self.source_text_idx_data, self.source_idx_length_data, self.source_sent_num_data,
                self.target_text_idx_data, self.target_idx_length_data
            )
        )
        random.shuffle(temp)
        self.knowledge_text_idx_data[:], self.knowledge_idx_length_data[:], self.knowledge_sent_num_data[:], \
        self.source_text_idx_data[:], self.source_idx_length_data[:], self.source_sent_num_data[:], \
        self.target_text_idx_data[:], self.target_idx_length_data[:] = zip(*temp)

    def _pad_batch_multi_sequence(self, text_idx_data, idx_length_data, sentence_num):
        max_sentence_num = max(sentence_num)
        max_length = max([max(sent_len) for sent_len in idx_length_data])

        new_length_data = []
        new_data = []
        for (text_idx, idx_len, sent_num) in zip(text_idx_data, idx_length_data, sentence_num):
            new_length_data.append(idx_len + [0] * (max_sentence_num - sent_num))
            new_sent_data = []
            for (sent_idx, idx_len) in zip(text_idx, idx_len):
                new_sent_data.append(sent_idx + [self.padding_token_idx] * (max_length - idx_len))
            for _ in range(max_sentence_num - sent_num):
                new_sent_data.append([0] * max_length)
            new_data.append(new_sent_data)
        sentence_num = torch.LongTensor(sentence_num)
        new_length_data = torch.LongTensor(new_length_data)
        new_data = torch.LongTensor(new_data)
        return new_data, new_length_data, sentence_num
        
    def _next_batch_data(self):
        knowledge_text = self.knowledge_text_data[self.pr:self.pr + self.step]
        tp_knowledge_text_idx_data = self.knowledge_text_idx_data[self.pr:self.pr + self.step]
        tp_knowledge_idx_length_data = self.knowledge_idx_length_data[self.pr:self.pr + self.step]
        tp_knowledge_sent_num = self.knowledge_sent_num_data[self.pr:self.pr + self.step]
        knowledge_idx, knowledge_length, knowledge_sentence_num = self._pad_batch_multi_sequence(tp_knowledge_text_idx_data, tp_knowledge_idx_length_data, tp_knowledge_sent_num)

        source_text = self.source_text_data[self.pr:self.pr + self.step]
        tp_source_text_idx_data = self.source_text_idx_data[self.pr:self.pr + self.step]
        tp_source_idx_length_data = self.source_idx_length_data[self.pr:self.pr + self.step]
        tp_source_sent_num = self.source_sent_num_data[self.pr:self.pr + self.step]
        source_idx, source_length, source_sentence_num = self._pad_batch_multi_sequence(tp_source_text_idx_data, tp_source_idx_length_data, tp_source_sent_num)

        target_text = self.target_text_data[self.pr:self.pr + self.step]
        tp_target_text_idx_data = self.target_text_idx_data[self.pr:self.pr + self.step]
        tp_target_idx_length_data = self.target_idx_length_data[self.pr:self.pr + self.step]
        target_idx, target_length = self._pad_batch_sequence(tp_target_text_idx_data, tp_target_idx_length_data)

        self.pr += self.step

        batch_data = {
            'knowledge_text': knowledge_text,
            'knowledge_idx': knowledge_idx.to(self.device),
            'knowledge_sentence_num': knowledge_sentence_num.to(self.device),
            'knowledge_length': knowledge_length.to(self.device),
            'source_text': source_text,
            'source_idx': source_idx.to(self.device),
            'source_sentence_num': source_sentence_num.to(self.device),
            'source_length': source_length.to(self.device),
            'target_text': target_text,
            'target_idx': target_idx.to(self.device),
            'target_length': target_length.to(self.device)
        }
        return batch_data
