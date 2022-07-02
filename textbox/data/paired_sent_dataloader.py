from textbox.data import AbstractDataLoader
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
            source_idx, source_length = pad_sequence(
                source_idx, source_length, self.padding_token_idx
            )

            batch_data = {
                'source_text': source_text,
                'source_idx': source_idx.to(self.device),
                'source_length': source_length.to(self.device)
            }
            return batch_data
        else:
            return {'source_text': source_text}
