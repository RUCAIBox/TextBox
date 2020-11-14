import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class BasicRNNEncoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 rnn_type,
                 bidirectional=True,
                 combine_method='concat'):
        super(BasicRNNEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.combine_method = combine_method

        if rnn_type == "lstm":
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == "gru":
            self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == "rnn":
            self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            print("error")

    def init_hidden(self, input_embeddings):
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_type == "gru":
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        elif self.rnn_type == "lstm":
            h_0 = torch.zeros(self.n_layers_encoder, batch_size, self.hidden_size)
            c_0 = torch.zeros(self.n_layers_encoder, batch_size, self.hidden_size)
            hidden_states = (h_0.to(device), c_0.to(device))
            return hidden_states
        else:
            raise NotImplementedError("No such initial hidden method for rnn type {}".format(self.rnn_type))

    def combine_states(self, step_states):
        if self.combine_method == 'concat':
            return step_states
        elif self.combine_method == 'sum':
            return step_states[:, :, :self.hidden_size] + step_states[:, : ,self.hidden_size:]
        else:
            return step_states[:, :, :self.hidden_size]

    def forward(self, input_embeddings, input_length, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        input_sort_idx = np.argsort(-np.array(input_length)).tolist()
        input_unsort_idx = np.argsort(input_sort_idx).tolist()

        input_length = np.array(input_length)[input_sort_idx].tolist()
        input_embeddings = input_embeddings[input_sort_idx, :, :]

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length)

        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[input_unsort_idx, :, :]

        if self.bidirectional:
            outputs = self.combine_states(outputs)
            return outputs, hidden_states

        return outputs, hidden_states
