# @Time   : 2020/11/11
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn


import torch
import torch.nn as nn

from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder


class MaliGANDiscriminator(BasicRNNDecoder):
    def __init__(self, embedding_size, hidden_size, num_layers, rnn_type, dropout_ratio):
        super(MaliGANGenerator, self).__init__(embedding_size, hidden_size, num_layers, rnn_type, dropout_ratio)
        self.hidden_linear = nn.Linear(num_layers * hidden_size, hidden_size)
        self.label_linear = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_embeddings, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        if self.rnn_type == 'lstm':
            _, (hidden, _) = self.decoder(input_embeddings, hidden_states)  # hidden: num_layers * batch_size * hidden_size
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            _, hidden = self.decoder(input_embeddings, hidden_states)  # hidden: num_layers * batch_size * hidden_size
        out = self.hidden_linear(hidden.view(-1, self.num_layers * self.hidden_size))  # batch_size * (num_layers * hidden_size) --> batch_size * hidden_size 
        pred = self.label_linear(self.dropout(torch.tanh(out)))  # batch_size * hidden_size --> batch_size * 2
        return pred