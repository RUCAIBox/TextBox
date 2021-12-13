# @Time   : 2021/10/12
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.kg_sent_dataloader
################################################
"""

from textbox.data.dataloader import AbstractDataLoader
from textbox.data.utils import pad_sequence


class KGSentenceDataLoader(AbstractDataLoader):
    r""":class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (KGSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
        super().__init__(config, dataset, batch_size, shuffle, drop_last, DDP)

    def _next_source_patch(self):
        source_text = self.source_text[self.pr:self.pr + self.step]
        source_idx = self.source_idx[self.pr:self.pr + self.step]
        source_length = self.source_length[self.pr:self.pr + self.step]
        source_triple = self.source_triple[self.pr:self.pr + self.step]
        source_triple_idx = self.source_triple_idx[self.pr:self.pr + self.step]
        source_entity = self.source_entity[self.pr:self.pr + self.step]
        target_mention = self.target_mention[self.pr:self.pr + self.step]
        dict_data = self.target_dict[self.pr:self.pr + self.step]
        source_idx, source_length, _ = pad_sequence(
            source_idx, source_length, self.padding_token_idx
        )

        batch_data = {
            'source_text': source_text,
            'source_idx': source_idx.to(self.device),
            'source_length': source_length.to(self.device),
            'source_triple': source_triple,
            'source_triple_idx': source_triple_idx,
            'source_entity': source_entity,
            'target_mention': target_mention,
            'target_dict': dict_data,
        }
        return batch_data
