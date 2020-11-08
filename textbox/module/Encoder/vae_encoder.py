import torch
from torch import nn
import torch.nn.functional as F

'''
Reference
https://github.com/rohithreddy024/VAE-Text-Generation/blob/master/model.py
'''


class Highway(nn.Module):
    def __init__(self, n_layers_highway, input_size):
        super(Highway, self).__init__()
        self.n_layers = n_layers_highway
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.n_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.n_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.n_layers)])

    def forward(self, x):
        for layer in range(self.n_layers):
            gate = F.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate*non_linear + (1-gate)*linear
            # Combine non linear and linear information according to gate
        return x


class LSTMVAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers_encoder, n_layers_highway):
        super(LSTMVAEEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers_encoder = n_layers_encoder
        self.highway = Highway(n_layers_highway=n_layers_highway, input_size=self.input_size)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.n_layers_encoder, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(2 * self.n_layers_encoder, batch_size, self.hidden_size)
        c_0 = torch.zeros(2 * self.n_layers_encoder, batch_size, self.hidden_size)
        hidden_states = (h_0.to(device), c_0.to(device))
        return hidden_states

    def forward(self, x):
        batch_size, n_seq, n_embed = x.size()
        x = self.highway(x)
        hidden_states = self.init_hidden(batch_size, device=x.device)
        _, (self.hidden, _) = self.lstm(x, hidden_states)
        # Exclude c_T and extract only h_T
        self.hidden = self.hidden.view(self.n_layers_encoder, 2, batch_size, self.hidden_size)
        self.hidden = self.hidden[-1]
        # Select only the final layer of h_T
        e_hidden = torch.cat(list(self.hidden), dim=1)
        # merge hidden states of both directions; check size
        return e_hidden


class BasicRNNEncoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 rnn_type,
                 token_embedder):
        super(BasicRNNEncoder, self).__init__()

        self.token_embedder = token_embedder
        self.vocab_linear = nn.Linear(hidden_size, output_size)

        if rnn_type == "lstm":
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "gru":
            self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "rnn":
            self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        else:
            print("error")

    def forward(self, hidden_states, input_seq):
        inputs = self.token_embedder(input_seq)
        outputs, hidden_states = self.encoder(inputs, hidden_states)
        return outputs, hidden_states
