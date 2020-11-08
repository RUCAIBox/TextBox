import torch
from torch import nn
import torch.nn.functional as F


class LSTMVAEDecoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers):
        super(LSTMVAEDecoder, self).__init__()

        self.vocab_linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def init_hidden(self, batch_size):
        h_0 = T.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = T.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden_states = (h_0.to(self.device), c_0.to(self.device))
        return hidden_states

    def forward(self, inputs, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(inputs.size(0))
        outputs, hidden_states = self.decoder(inputs, hidden_states)
        # print(outputs.size(), hidden_states.size())
        token_logits = self.vocab_linear(outputs)
        # print(token_logits.size())
        return token_logits, hidden_states

