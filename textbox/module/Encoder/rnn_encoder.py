# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class BasicRNNEncoder(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_enc_layers,
                 rnn_type,
                 dropout_ratio,
                 bidirectional=True):
        super(BasicRNNEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_enc_layers = num_enc_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        if rnn_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_enc_layers,
                                   batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_enc_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_enc_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_enc_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_enc_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            return torch.zeros(self.num_enc_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing encoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, input_length, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length,
                                                                          batch_first=True, enforce_sorted=False)

        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden_states
