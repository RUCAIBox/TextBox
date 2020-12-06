# @Time   : 2020/11/16
# @Author : Junyi Li
# @email  : lijunyie@ruc.edu.cn

"""
textbox.data.dataloader.paired_sent_dataloader
################################################
"""

import numpy as np
import random
import math
import pandas as pd
import torch
from tqdm import tqdm

from textbox.data.dataloader.abstract_dataloader import AbstractDataLoader
from textbox.utils import DataLoaderType, InputType


class PairedSentenceDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (SingleSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~textbox.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, shuffle=shuffle)
        self.data_preprocess(dataset)

    def data_preprocess(self, dataset):
        required_key_list = ['source_text_data', 'target_text_data', 'source_token2idx',
                             'source_idx2token', 'target_token2idx', 'target_idx2token']
        for dataset_attr in required_key_list:
            assert dataset_attr in dataset
            setattr(self, dataset_attr, dataset[dataset_attr])
        self.source_text_idx_data, self.source_idx_length_data = self._build_data(self.source_text_data,
                                                                                  self.source_token2idx)
        self.target_text_idx_data, self.target_idx_length_data = self._build_data(self.target_text_data,
                                                                                  self.target_token2idx)

    def get_reference(self):
        return self.target_text_data

    @property
    def pr_end(self):
        return len(self.target_text_idx_data)

    def __len__(self):
        return math.ceil(len(self.target_text_idx_data) / self.batch_size)

    def _shuffle(self):
        temp = list(zip(self.source_text_data, self.source_text_idx_data, self.source_idx_length_data,
                        self.target_text_data, self.target_text_idx_data, self.target_idx_length_data))
        random.shuffle(temp)
        self.source_text_data[:], self.source_text_idx_data[:], self.source_idx_length_data[:], \
        self.target_text_data[:], self.target_text_idx_data[:], self.target_idx_length_data[:] = zip(*temp)

    def _next_batch_data(self):
        source_text = self.source_text_data[self.pr: self.pr + self.step]
        tp_source_text_idx_data = self.source_text_idx_data[self.pr: self.pr + self.step]
        tp_source_idx_length_data = self.source_idx_length_data[self.pr: self.pr + self.step]
        source_idx, source_length = self._pad_batch_sequence(tp_source_text_idx_data, tp_source_idx_length_data)

        target_text = self.target_text_data[self.pr: self.pr + self.step]
        tp_target_text_idx_data = self.target_text_idx_data[self.pr: self.pr + self.step]
        tp_target_idx_length_data = self.target_idx_length_data[self.pr: self.pr + self.step]
        target_idx, target_length = self._pad_batch_sequence(tp_target_text_idx_data, tp_target_idx_length_data)

        self.pr += self.step

        batch_data = {
            'source_text': source_text,
            'source_idx': source_idx.to(self.device),
            'source_length': source_length.to(self.device),
            'target_text': target_text,
            'target_idx': target_idx.to(self.device),
            'target_length': target_length.to(self.device)
        }
        return batch_data

