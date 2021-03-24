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
        dataset (MultipleSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, shuffle=shuffle)
        self.group_data_key = []
        self.data_preprocess(dataset)

    def _build_multi_data(self, text_data, token2idx, need_text_start_end=True):
        r"""transform text to id and add sos and eos token index.
        input:
            text_data: list -> list -> list -> character, original text
            token2idx: dict, map token to index
            need_text_start_end, bool, indicates whether we should add sos and eos token index.
        output:
            text_idx_data: list -> list -> list -> int, list of word index
            idx_length_data: list -> list -> int, list of sequence length
            idx_num_data: list of sentence number
        """
        text_idx_data = []
        idx_length_data = []
        idx_num_data = []
        for text in text_data:
            idx_data = []
            idx_length = []
            for sent in text:
                text_idx = self._token2idx(sent, token2idx)
                if need_text_start_end:
                    text_idx = [self.sos_token_idx] + text_idx + [self.eos_token_idx]
                idx_data.append(text_idx)
                idx_length.append(len(text_idx))
            text_idx_data.append(idx_data)
            idx_length_data.append(idx_length)
            idx_num_data.append(len(text))
        return text_idx_data, idx_length_data, idx_num_data

    def data_preprocess(self, dataset):
        required_key_list = ['idx2token', 'token2idx']
        for dataset_attr in required_key_list:
            assert dataset_attr in dataset
            setattr(self, dataset_attr, dataset[dataset_attr])
        for group in ['knowledge', 'source', 'target']:
            text_name = group + '_text_data'
            if text_name in dataset:
                text_data = dataset[text_name]
                setattr(self, text_name, text_data)
                if isinstance(text_data[0][0], str):
                    text_idx, idx_length = self._build_data(text_data, self.token2idx)
                else:
                    text_idx, idx_length, idx_num = self._build_multi_data(text_data, self.token2idx)
                    setattr(self, group + '_idx_num_data', idx_num)
                    self.group_data_key.append(group + '_idx_num_data')
                setattr(self, group + '_text_idx_data', text_idx)
                self.group_data_key.append(group + '_text_idx_data')
                setattr(self, group + '_idx_length_data', idx_length)
                self.group_data_key.append(group + '_idx_length_data')

    def get_reference(self):
        return self.target_text_data

    @property
    def pr_end(self):
        return len(self.target_text_idx_data)

    def __len__(self):
        return math.ceil(len(self.target_text_idx_data) / self.batch_size)

    def _shuffle(self):
        data_list = []
        for key in self.group_data_key:
            data_list.append(getattr(self, key))
        temp = list(zip(*data_list))
        random.shuffle(temp)
        temp = list(zip(*temp))
        for i, key in enumerate(self.group_data_key):
            setattr(self, key, list(temp[i]))

    def _pad_batch_multi_sequence(self, text_idx_data, idx_length_data, idx_num_data):
        max_num = max(idx_num_data)
        max_length = max([max(length) for length in idx_length_data])

        new_length_data = []
        new_idx_data = []
        for (text_idx, idx_len, idx_num) in zip(text_idx_data, idx_length_data, idx_num_data):
            new_length_data.append(idx_len + [0] * (max_num - idx_num))
            new_sent_data = []
            for (sent_idx, idx_len) in zip(text_idx, idx_len):
                new_sent_data.append(sent_idx + [self.padding_token_idx] * (max_length - idx_len))
            for _ in range(max_num - idx_num):
                new_sent_data.append([0] * max_length)
            new_idx_data.append(new_sent_data)
        new_num_data = torch.LongTensor(idx_num_data)
        new_length_data = torch.LongTensor(new_length_data)
        new_idx_data = torch.LongTensor(new_idx_data)
        return new_idx_data, new_length_data, new_num_data

    def _next_batch_data(self):
        batch_data = {}
        for group in ['knowledge', 'source', 'target']:
            text_name = group + '_text_data'
            if hasattr(self, text_name):
                batch_data[text_name] = getattr(self, text_name)[self.pr:self.pr + self.step]

                idx_name = group + '_text_idx_data'
                length_name = group + '_idx_length_data'
                num_name = group + '_idx_num_data'
                tp_text_idx = getattr(self, idx_name)[self.pr:self.pr + self.step]
                tp_idx_length = getattr(self, length_name)[self.pr:self.pr + self.step]
                if hasattr(self, num_name):
                    tp_idx_num = getattr(self, num_name)[self.pr:self.pr + self.step]
                    text_idx, idx_length, idx_num = self._pad_batch_multi_sequence(
                        tp_text_idx, tp_idx_length, tp_idx_num
                    )
                    batch_data[num_name] = idx_num.to(self.device)
                else:
                    text_idx, idx_length = self._pad_batch_sequence(tp_text_idx, tp_idx_length)

                batch_data[idx_name] = text_idx.to(self.device)
                batch_data[length_name] = idx_length.to(self.device)

        self.pr += self.batch_size
        return batch_data
