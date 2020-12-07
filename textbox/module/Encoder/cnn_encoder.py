import torch
from torch import nn
import torch.nn.functional as F

'''
Reference
https://github.com/rohithreddy024/VAE-Text-Generation/blob/master/model.py
'''


class BasicCNNEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(BasicCNNEncoder, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size

        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 128, 3, 1),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(128, 256, 3, 1),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 256, 3, 1),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 512, 3, 1),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, self.latent_size, 3, 1),
            nn.BatchNorm1d(self.latent_size),
            nn.ELU()
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
        :return: An float tensor with shape of [batch_size, latent_variable_size]
        """
        input = input.transpose(1, 2).contiguous()
        output = self.cnn(input)
        output = torch.mean(output, dim=-1)
        return output
