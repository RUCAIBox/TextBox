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

        if rnn_type == "lstm":
            self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "gru":
            self.decoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "rnn":
            self.decoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        else:
            print("error")

    def forward(self, hidden_states, input_embeddings):
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
