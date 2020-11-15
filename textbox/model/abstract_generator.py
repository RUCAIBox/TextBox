# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn


"""
recbole.model.abstract_recommender
##################################
"""

import numpy as np
import torch
import torch.nn as nn

from textbox.utils import ModelType, InputType, FeatureSource, FeatureType


class AbstractGenerator(nn.Module):
    r"""Base class for all models
    """

    def calculate_loss(self, corpus):
        r"""Calculate the training loss for a batch data.

        Args:
            corpus (Corpus): Corpus class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def generate(self, corpus):
        r"""Predict the scores between users and items.

        Args:
            corpus (Corpus): Corpus class of the batch.

        Returns:
            torch.Tensor: Generated text, shape: [batch_size, max_len]
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class UnconditionalGenerator(AbstractGenerator):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    type = ModelType.UNCONDITIONAL

    def __init__(self, config, dataset):
        super(AbstractGenerator, self).__init__()

        self.vocab_size = len(dataset.idx2token)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']


class ConditionalGenerator(AbstractGenerator):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    type = ModelType.CONDITIONAL

    def __init__(self, config, dataset):
        super(AbstractGenerator, self).__init__()

        self.vocab_size = len(dataset.idx2token)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']



