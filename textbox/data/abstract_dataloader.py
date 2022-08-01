import math
import torch
import random
from logging import getLogger
from textbox.data.misc import _pad_sequence
from torch.nn.utils.rnn import pad_sequence



class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data.
        And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Corpus): The corpus for partition of dataset.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.

    Attributes:
        dataset (dict): The necessary elements of this dataloader.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False, drop_last=True):
        self.config = config
        self.device = config['device']
        self.logger = getLogger(__name__)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.step = batch_size
        self.pr = 0

        self.std_pr = 0
        self.pr_end = len(self.target_text)

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        return None

    def __len__(self):
        return math.floor(self.pr_end / self.batch_size) if self.drop_last else math.ceil(self.pr_end / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if (self.drop_last
            and self.std_pr + self.batch_size >= self.pr_end) or (not self.drop_last and self.pr >= self.pr_end):
            self.pr = 0
            self.std_pr = 0
            raise StopIteration()

        next_batch = self._next_batch_data()
        self.pr += self.batch_size
        self.std_pr += self.batch_size
        return next_batch

    def _shuffle(self):
        r"""Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        keys = []
        values = []
        for key, value in self.dataset.__dict__.items():
            if key.startswith(('source', 'target')) and isinstance(value,
                                                                   list) and isinstance(value[0], (list, str, int)):
                keys.append(key)
                values.append(value)
        values = list(zip(*values))
        random.shuffle(values)
        for key, value in zip(keys, list(zip(*values))):
            getattr(self.dataset, key)[:] = value

    def _next_source_patch(self):
        r"""Assemble next batch of source data in form of Interaction, and return these data.
        
        Returns:
            Interaction: The next batch of source data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

    def _next_target_patch(self):
        r"""Assemble next batch of target data in form of Interaction, and return these data.
        
        Returns:
            Interaction: The next batch of target data.
        """
        target_text = self.target_text[self.pr:self.pr + self.step]
        if self.target_idx is not None:
            target_idx = self.target_idx[self.pr:self.pr + self.step]
            target_length = self.target_length[self.pr:self.pr + self.step]
            target_idx, target_length = pad_sequence(
                target_idx, target_length, self.padding_token_idx
            )

            batch_data = {
                'target_text': target_text,
                'target_idx': target_idx.to(self.device),
                'target_length': target_length.to(self.device)
            }
            return batch_data
        else:
            return {'target_text': target_text}

    def _next_batch_data(self):
        r"""Assemble next batch of data in form of Interaction, and return these data.
        
        Returns:
            Interaction: The next batch of data.
        """
        source_batch = self._next_source_patch()
        target_batch = self._next_target_patch()
        return dict(**source_batch, **target_batch)

    def get_reference(self):
        r"""Get reference documents for current data loader
        return is supposed to be reference_corpus as list -> list -> word
        """
        target_text = self.target_text if isinstance(self.target_text[0][0], str) else [sum(doc, []) for doc in self.target_text]
        if self.config['tokenize_strategy'] == 'none':
            return [text.split(' ') for text in target_text]
        else:
            return target_text
