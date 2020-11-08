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
            :obj:`~textbox.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    dl_type = DataLoaderType.UNCONDITIONAL

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.NOISE, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)
        self.padding_index = 0  # config['padding_index']

    def setup(self):
        pass

    def data_preprocess(self, dataset):
        # attr_list = ['idx2token', 'reference_corpus', 'target_text_idx_data', 'target_idx_length_data']
        # for dataset_attr in attr_list:
        #     setattr(self, dataset_attr, getattr(dataset, dataset_attr))
        self.idx2token = dataset.idx2token
        self.reference_corpus = dataset.target_text_data
        self.target_text_idx_data = dataset.target_text_idx_data
        self.target_idx_length_data = dataset.target_idx_length_data

    def get_reference(self):
        return self.reference_corpus

    @property
    def pr_end(self):
        return len(self.target_text_idx_data)

    def __len__(self):
        return math.ceil(len(self.target_text_idx_data) / self.batch_size)

    def _shuffle(self):
        temp = list(zip(self.target_text_idx_data, self.target_idx_length_data))
        random.shuffle(temp)
        self.target_text_idx_data[:], self.target_idx_length_data[:] = zip(*temp)

    def _next_batch_data(self):
        tp_text_idx_data = self.target_text_idx_data[self.pr: self.pr + self.step]
        tp_idx_length_data = self.target_idx_length_data[self.pr: self.pr + self.step]
        self.pr += self.step
        padded_text, length = self._pad_batch_sequence(tp_text_idx_data, tp_idx_length_data)
        batch_data = {
            'target_text': padded_text.to(self.device),
            'target_length': length.to(self.device)
        }
        return batch_data

