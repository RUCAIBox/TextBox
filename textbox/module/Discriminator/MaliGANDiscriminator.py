# @Time   : 2020/11/17
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn

r"""
MaliGAN Discriminator
#####################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from textbox.model.abstract_generator import UnconditionalGenerator


class MaliGANDiscriminator(UnconditionalGenerator):
    r"""MaliGANDiscriminator is LSTMs.
    """

    def __init__(self, config, dataset):
        super(MaliGANDiscriminator, self).__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.embedding_size = config['discriminator_embedding_size']
        self.max_length = config['max_seq_length'] + 2
        self.num_dis_layers = config['num_dis_layers']
        self.dropout_rate = config['dropout_rate']

        self.pad_idx = dataset.padding_token_idx

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.num_dis_layers, batch_first=True)
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.pad_idx)
        self.vocab_projection = nn.Linear(self.hidden_size, self.vocab_size)

        self.hidden_linear = nn.Linear(self.num_dis_layers * self.hidden_size, self.hidden_size)
        self.label_linear = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, data):
        r"""Calculate the probability that the data is realistic.

        Args:
            data (torch.Tensor): The sentence data, shape: [batch_size, max_seq_len].

        Returns:
            torch.Tensor: The probability that each sentence is realistic, shape: [batch_size].
        """
        data_embedding = self.word_embedding(data)  # b * l * e
        _, (hidden, _) = self.LSTM(data_embedding)  # hidden: b * num_layers * h
        out = self.hidden_linear(
            hidden.view(-1, self.num_dis_layers * self.hidden_size)
        )  # b * (num_layers * h) -> b * h
        pred = self.label_linear(self.dropout(torch.tanh(out))).squeeze(1)  # b * h -> b
        pred = torch.sigmoid(pred)
        return pred

    def calculate_loss(self, real_data, fake_data):
        r"""Calculate the loss for real data and fake data.
        The discriminator is trained with the standard objective that GAN employs.

        Args:
            real_data (torch.Tensor): The realistic sentence data, shape: [batch_size, max_seq_len].
            fake_data (torch.Tensor): The generated sentence data, shape: [batch_size, max_seq_len].

        Returns:
            torch.Tensor: The calculated loss of real data and fake data, shape: [].
        """
        real_y = self.forward(real_data)  # b * l --> b
        fake_y = self.forward(fake_data)
        logits = torch.cat((real_y, fake_y), dim=0)

        real_label = torch.ones_like(real_y)
        fake_label = torch.zeros_like(fake_y)
        target = torch.cat((real_label, fake_label), dim=0)

        loss = F.binary_cross_entropy(logits, target)
        return loss
