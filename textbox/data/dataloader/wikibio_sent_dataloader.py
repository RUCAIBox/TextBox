# @Time   : 2021/10/12
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.wikibio_sent_dataloader
################################################
"""

from textbox.data.dataloader import AbstractDataLoader
from textbox.data.utils import pad_sequence


class WikiBioSentenceDataLoader(AbstractDataLoader):
    r""":class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (WikiBioSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
        super().__init__(config, dataset, batch_size, shuffle, drop_last, DDP)

    def _next_source_patch(self):
        source_key_text = self.source_key_text[self.pr:self.pr + self.step]
        source_key_idx = self.source_key_idx[self.pr:self.pr + self.step]
        source_value_text = self.source_value_text[self.pr:self.pr + self.step]
        source_value_idx = self.source_value_idx[self.pr:self.pr + self.step]

        batch_data = {
            'source_key_text': source_key_text,
            'source_key_idx': source_key_idx,
            'source_value_text': source_value_text,
            'source_value_idx': source_value_idx,
        }
        return batch_data
