# @Time   : 2021/1/30
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/10
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.attr_sent_dataloader
################################################
"""

import torch
from textbox.data.dataloader import AbstractDataLoader


class AttributedSentenceDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (AttributedSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
        super().__init__(config, dataset, batch_size, shuffle, drop_last, DDP)
        self.attribute_size = [len(a2t) for a2t in self.source_idx2token]
        self.attribute_num = len(self.attribute_size)

    def _next_source_patch(self):
        source_text = self.source_text[self.pr:self.pr + self.step]
        source_idx = self.source_idx[self.pr:self.pr + self.step]
        source_idx = torch.LongTensor(source_idx)

        batch_data = {
            'source_text': source_text,
            'source_idx': source_idx.to(self.device),
        }
        return batch_data
