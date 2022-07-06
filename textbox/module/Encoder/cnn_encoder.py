# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

r"""
CNN Encoder
############
"""

import torch
from torch import nn
import torch.nn.functional as F


class BasicCNNEncoder(nn.Module):
    r"""
    Basic Convolution Neural Network (CNN) encoder.
    Code reference: https://github.com/rohithreddy024/VAE-Text-Generation/
    """

    def __init__(self, input_size, latent_size):
        super(BasicCNNEncoder, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size

        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 128, 3, 1), nn.BatchNorm1d(128), nn.ELU(), nn.Conv1d(128, 256, 3, 1),
            nn.BatchNorm1d(256), nn.ELU(), nn.Conv1d(256, 256, 3, 1), nn.BatchNorm1d(256), nn.ELU(),
            nn.Conv1d(256, 512, 3, 1), nn.BatchNorm1d(512), nn.ELU(), nn.Conv1d(512, self.latent_size, 3, 1),
            nn.BatchNorm1d(self.latent_size), nn.ELU()
        )

    def forward(self, input):
        r""" Implement the encoding process.

        Args:
            input (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            torch.Tensor: output features, shape: [batch_size, sequence_length, feature_size].
        """
        input = input.transpose(1, 2).contiguous()
        output = self.cnn(input)
        output = torch.mean(output, dim=-1)
        return output
