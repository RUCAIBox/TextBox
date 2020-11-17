# @Time   : 2020/11/4
# @Author : Gaole He
# @email  : hegaole@ruc.edu.cn

"""
textbox.data.dataloader.general_dataloader
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


class SingleSentenceDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (SingleSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~textbox.utils.enum_type.InputType.NOISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    dl_type = DataLoaderType.UNCONDITIONAL

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.NOISE, shuffle=False):

        super().__init__(config, dataset, batch_size, dl_format, shuffle)
        self.data_preprocess(dataset)

    def data_preprocess(self, dataset):
        required_key_list = ['text_data']
        for dataset_attr in required_key_list:
            assert dataset_attr in dataset
            setattr(self, dataset_attr, dataset[dataset_attr])
        self.text_idx_data, self.idx_length_data = self._build_data(self.text_data, self.token2idx)

    def get_reference(self):
        return self.text_data

    @property
    def pr_end(self):
        return len(self.text_idx_data)

    @property
    def vocab_size(self):
        return len(self.idx2token)

    def __len__(self):
        return math.ceil(len(self.text_idx_data) / self.batch_size)

    def _shuffle(self):
        temp = list(zip(self.text_data, self.text_idx_data, self.idx_length_data))
        random.shuffle(temp)
        self.text_data[:], self.text_idx_data[:], self.idx_length_data[:] = zip(*temp)

    def _next_batch_data(self):
        tp_text_data = self.text_data[self.pr: self.pr + self.step]
        tp_text_idx_data = self.text_idx_data[self.pr: self.pr + self.step]
        tp_idx_length_data = self.idx_length_data[self.pr: self.pr + self.step]
        self.pr += self.step
        padded_idx, length = self._pad_batch_sequence(tp_text_idx_data, tp_idx_length_data)
        batch_data = {
            'target_text': tp_text_data,
            'target_idx': padded_idx.to(self.device),
            'target_length': length.to(self.device)
        }
        return batch_data

