# @Time   : 2021/10/12
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.rotowire_sent_dataloader
################################################
"""

import torch
from textbox.data.dataloader import AbstractDataLoader
from textbox.data.utils import pad_sequence


class RotoWireSentenceDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (RotoWireSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
        super().__init__(config, dataset, batch_size, shuffle, drop_last, DDP)

    def _next_source_patch(self):
        source_text = self.source_text[self.pr:self.pr + self.step]
        source_idx = self.source_idx[self.pr:self.pr + self.step]
        source_length = self.source_length[self.pr:self.pr + self.step]
        tmp_length = [[4] * l for l in source_length]
        source_idx, _, source_length = pad_sequence(
            source_idx, tmp_length, self.padding_token_idx, source_length
        )

        batch_data = {
            'source_text': source_text,
            'source_idx': source_idx.to(self.device),
            'source_length': source_length.to(self.device)
        }

        if hasattr(self, 'source_plan_idx'):
            source_plan_idx = self.source_plan_idx[self.pr:self.pr + self.step]
            source_plan_length = self.source_plan_length[self.pr:self.pr + self.step]
            source_plan_idx, source_plan_length, _ = pad_sequence(
                source_plan_idx, source_plan_length, self.padding_token_idx, None
            )
            batch_data['source_plan_idx'] = source_plan_idx.to(self.device)
            batch_data['source_plan_length'] = source_plan_length.to(self.device)
        return batch_data
