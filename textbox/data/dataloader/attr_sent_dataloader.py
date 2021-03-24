# @Time   : 2021/1/30
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.attr_sent_dataloader
################################################
"""

import numpy as np
import random
import math
import torch

from textbox.data.dataloader.abstract_dataloader import AbstractDataLoader


class AttributedSentenceDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (AttributedSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, batch_size, shuffle)
        self._data_preprocess(dataset)

    def _build_attribute(self, attribute_data, attribute2idx):
        attribute_idx_data = []
        for attribute in attribute_data:
            attribute_idx = []
            for i, attr in enumerate(attribute):
                idx = attribute2idx[i][attr]
                attribute_idx.append(idx)
            attribute_idx_data.append(attribute_idx)
        return attribute_idx_data

    def _data_preprocess(self, dataset):
        required_key_list = ['text_data', 'idx2token', 'token2idx', 'attribute_data', 'idx2attribute', 'attribute2idx']
        for dataset_attr in required_key_list:
            assert dataset_attr in dataset
            setattr(self, dataset_attr, dataset[dataset_attr])
        self.text_idx_data, self.idx_length_data = self._build_data(self.text_data, self.token2idx)
        self.attribute_idx_data = self._build_attribute(self.attribute_data, self.attribute2idx)

    def get_reference(self):
        return self.text_data

    @property
    def pr_end(self):
        return len(self.text_idx_data)

    def __len__(self):
        return math.ceil(len(self.text_idx_data) / self.batch_size)

    def _shuffle(self):
        temp = list(
            zip(self.text_data, self.text_idx_data, self.idx_length_data, self.attribute_data, self.attribute_idx_data)
        )
        random.shuffle(temp)
        self.text_data[:
                       ], self.text_idx_data[:
                                             ], self.idx_length_data[:
                                                                     ], self.attribute_data[:
                                                                                            ], self.attribute_idx_data[:] = zip(
                                                                                                *temp
                                                                                            )

    def _next_batch_data(self):
        tp_text_data = self.text_data[self.pr:self.pr + self.step]
        tp_text_idx_data = self.text_idx_data[self.pr:self.pr + self.step]
        tp_idx_length_data = self.idx_length_data[self.pr:self.pr + self.step]
        padded_idx, length = self._pad_batch_sequence(tp_text_idx_data, tp_idx_length_data)

        tp_attribute_data = self.attribute_data[self.pr:self.pr + self.step]
        tp_attribute_idx_data = self.attribute_idx_data[self.pr:self.pr + self.step]
        attribute_idx = torch.LongTensor(tp_attribute_idx_data)

        self.pr += self.batch_size

        batch_data = {
            'target_text': tp_text_data,
            'target_idx': padded_idx.to(self.device),
            'target_length': length.to(self.device),
            'attribute_text': tp_attribute_data,
            'attribute_idx': attribute_idx.to(self.device)
        }
        return batch_data
