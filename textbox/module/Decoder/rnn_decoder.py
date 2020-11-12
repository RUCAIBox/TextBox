import torch
from torch import nn
import torch.nn.functional as F


class BasicRNNDecoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 rnn_type):
        super(BasicRNNDecoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        if rnn_type == "lstm":
            self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "gru":
            self.decoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "rnn":
            self.decoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
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

    def forward(self, input_embeddings, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        outputs, hidden_states = self.decoder(input_embeddings, hidden_states)
        return outputs, hidden_states


class AttentionalRNNDecoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 encoder_output_size,
                 num_layers,
                 rnn_type,
                 attn_type,
                 token_embedder):
        super(AttentionalRNNDecoder, self).__init__()

        self.token_embedder = token_embedder
        self.vocab_linear = nn.Linear(hidden_size, output_size)

        if rnn_type == "lstm":
            self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "gru":
            self.decoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "rnn":
            self.decoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        else:
            print("error")

        # if attn_type == "luong":
        #     self.attentioner = LuongAttention()

    def forward(self, hidden_states, input_seq, encoder_outputs):
        inputs = self.token_embedder(input_seq)
        outputs, hidden_states = self.decoder(inputs, hidden_states)
        # context_outputs = self.attentioner(outputs, encoder_outputs)

        token_logits = self.vocab_linear(outputs)

        return token_logits, hidden_states
