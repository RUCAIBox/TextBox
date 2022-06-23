# @Time   : 2020/11/16
# @Author : Junyi Li
# @email  : lijunyie@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/10, 2021/1/29
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.paired_sent_dataloader
################################################
"""

from textbox.data.dataloader import AbstractDataLoader
from textbox.data.utils import pad_sequence


class PairedSentenceDataLoader(AbstractDataLoader):
    r""":class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (PairedSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
        super().__init__(config, dataset, batch_size, shuffle, drop_last, DDP)

    def _next_source_patch(self):
        source_text = self.source_text[self.pr:self.pr + self.step]
        if self.source_idx is not None:
            source_idx = self.source_idx[self.pr:self.pr + self.step]
            source_length = self.source_length[self.pr:self.pr + self.step]
            source_num = self.source_num[self.pr:self.pr + self.step] if self.source_num is not None else None
            source_idx, source_length, source_num = pad_sequence(
                source_idx, source_length, self.padding_token_idx, source_num
            )

            batch_data = {
                'source_text': source_text,
                'source_idx': source_idx.to(self.device),
                'source_length': source_length.to(self.device)
            }
            if source_num is not None:
                batch_data['source_num'] = source_num
            return batch_data
        else:
            return {'source_text': source_text}


class CopyPairedSentenceDataLoader(PairedSentenceDataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
        super().__init__(config, dataset, batch_size, shuffle, drop_last, DDP)

    def _next_source_patch(self):
        batch_data = super()._next_source_patch()  # source_text & source_idx & source_length
        if self.config['is_pgen']:
            source_extended_idx = self.source_extended_idx[self.pr:self.pr + self.step]
            source_extended_idx, _, _ = pad_sequence(
                source_extended_idx, batch_data['source_length'].cpu(), self.padding_token_idx)
            source_oovs = self.source_oovs[self.pr:self.pr + self.step]
            extra_zeros = self.get_extra_zeros(source_oovs)

            batch_data['source_extended_idx'] = source_extended_idx.to(self.device)
            batch_data['source_oovs'] = source_oovs
            batch_data['extra_zeros'] = extra_zeros.to(self.device)

        return batch_data

    def _next_target_patch(self):
        batch_data = super()._next_target_patch()  # target_text
        target_input_idx = self.target_input_idx[self.pr:self.pr + self.step]
        target_output_idx = self.target_output_idx[self.pr:self.pr + self.step]
        target_length = self.target_length[self.pr:self.pr + self.step]

        target_input_idx, target_length, _ = pad_sequence(target_input_idx, target_length, self.padding_token_idx)
        target_output_idx, _, _ = pad_sequence(target_output_idx, target_length, self.padding_token_idx)

        batch_data['target_input_idx'] = target_input_idx.to(self.device)
        batch_data['target_output_idx'] = target_output_idx.to(self.device)
        batch_data['target_length'] = target_length.to(self.device)

        return batch_data

    @staticmethod
    def get_extra_zeros(oovs):
        import torch
        max_oovs_num = max([len(_) for _ in oovs])
        extra_zeros = torch.zeros(len(oovs), max_oovs_num)
        return extra_zeros

