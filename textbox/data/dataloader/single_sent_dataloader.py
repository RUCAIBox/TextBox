# @Time   : 2020/11/4
# @Author : Gaole He
# @email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/10, 2021/1/29
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataloader.single_sent_dataloader
################################################
"""

from textbox.data.dataloader import AbstractDataLoader


class SingleSentenceDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (SingleSentenceDataset): The dataset of dataloader. Corpus, see textbox.data.corpus for more details
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True, DDP=False):
        super().__init__(config, dataset, batch_size, shuffle, drop_last, DDP)

    def _next_source_patch(self):
        return dict()
