# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn


class AbstractGAN(nn.Module):
    r"""Base class for GAN models
    """

    def __init__(self, config, dataset):
        super(AbstractGAN, self).__init__()

        self.vocab_size = len(dataset.idx2token)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

    def calculate_loss_G(self, corpus):
        r"""Calculate the training loss for generator.

        Args:
            corpus (Corpus): Corpus class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def calculate_loss_D(self, corpus):
        r"""Calculate the training loss for discriminator.

        Args:
            corpus (Corpus): Corpus class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def generate(self, corpus):
        r"""Generate text with trained generator model.

        Args:
            corpus (Corpus): Corpus class of the batch.

        Returns:
            torch.Tensor: Generated text, shape: [batch_size, max_len]
        """
        raise NotImplementedError
