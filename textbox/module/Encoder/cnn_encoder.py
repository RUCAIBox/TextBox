import torch
from torch import nn
import torch.nn.functional as F

'''
Reference
https://github.com/rohithreddy024/VAE-Text-Generation/blob/master/model.py
'''


class HybridEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(HybridEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 128, 4, 2),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(128, 256, 4, 2),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 256, 4, 2),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 512, 4, 2),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, self.hidden_size, 4, 2),
            nn.BatchNorm1d(self.hidden_size),
            nn.ELU()
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
        :return: An float tensor with shape of [batch_size, latent_variable_size]
        """
        input = input.permute(0, 2, 1)
        result = self.cnn(input)
        return result.squeeze(2)
