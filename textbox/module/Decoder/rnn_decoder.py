# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
from torch import nn
import torch.nn.functional as F
from textbox.module.Attention.attention_mechanism import LuongAttention, BahdanauAttention, MonotonicAttention
from textbox.module.Attention.attention_mechanism import LuongMonotonicAttention, BahdanauMonotonicAttention


class BasicRNNDecoder(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_dec_layers,
                 rnn_type,
                 dropout_ratio=0.0):
        super(BasicRNNDecoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_dec_layers = num_dec_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == "gru":
            self.decoder = nn.GRU(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == "rnn":
            self.decoder = nn.RNN(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("The RNN type in decoder must in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        outputs, hidden_states = self.decoder(input_embeddings, hidden_states)
        return outputs, hidden_states


class AttentionalRNNDecoder(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 context_size,
                 num_dec_layers,
                 rnn_type,
                 dropout_ratio=0.0,
                 attention_type=''):
        super(AttentionalRNNDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_type = rnn_type
        self.attention_type = attention_type

        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'gru':
            self.decoder = nn.GRU(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'rnn':
            self.decoder = nn.RNN(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("The RNN type in attentional decoder must in ['lstm', 'gru', 'rnn'].")

        if attention_type == 'LuongAttention':
            self.attentioner = LuongAttention()
        elif attention_type == 'BahdanauAttention':
            self.attentioner = BahdanauAttention()
        elif attention_type == 'MonotonicAttention':
            self.attentioner = MonotonicAttention()
        elif attention_type == 'LuongMonotonicAttention':
            self.attentioner = LuongMonotonicAttention()
        elif attention_type == 'BahdanauMonotonicAttention':
            self.attentioner = BahdanauMonotonicAttention()
        else:
            raise ValueError("The attention type in attentional decoder must in ['LuongAttention', 'BahdanauAttention'"
                             ", 'MonotonicAttention', 'LuongMonotonicAttention', 'BahdanauMonotonicAttention'].")

        self.attention_dense = nn.Linear(hidden_size + context_size, hidden_size)

    def init_hidden(self, input_embeddings):
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, hidden_states=None, encoder_outputs=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        outputs, hidden_states = self.decoder(inputs, hidden_states)
        contexts = self.attentioner(outputs, encoder_outputs)

        outputs = self.attention_dense(torch.cat([outputs, contexts]))

        return outputs, hidden_states
