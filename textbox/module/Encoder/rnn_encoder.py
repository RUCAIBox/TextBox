# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/9/1
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com

r"""
RNN Encoder
############
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from textbox.module.layers import LstmUnit, FieldGateLstmUnit


class BasicRNNEncoder(torch.nn.Module):
    r"""
    Basic Recurrent Neural Network (RNN) encoder.
    """

    def __init__(self, embedding_size, hidden_size, num_enc_layers, rnn_type, dropout_ratio, bidirectional=True):
        super(BasicRNNEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_enc_layers = num_enc_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        if rnn_type == 'lstm':
            self.encoder = nn.LSTM(
                embedding_size,
                hidden_size,
                num_enc_layers,
                batch_first=True,
                dropout=dropout_ratio,
                bidirectional=bidirectional
            )
        elif rnn_type == 'gru':
            self.encoder = nn.GRU(
                embedding_size,
                hidden_size,
                num_enc_layers,
                batch_first=True,
                dropout=dropout_ratio,
                bidirectional=bidirectional
            )
        elif rnn_type == 'rnn':
            self.encoder = nn.RNN(
                embedding_size,
                hidden_size,
                num_enc_layers,
                batch_first=True,
                dropout=dropout_ratio,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_enc_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_enc_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            tp_vec = torch.zeros(self.num_enc_layers * self.num_directions, batch_size, self.hidden_size)
            return tp_vec.to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing encoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, input_length, hidden_states=None):
        r""" Implement the encoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            input_length (Torch.Tensor): length of input sequence, shape: [batch_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [num_layers * num_directions, batch_size, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input_embeddings, input_length.cpu(), batch_first=True, enforce_sorted=False
        )

        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden_states


class NonFieldLstmEncoder(torch.nn.Module):
    r"""
    Lstm Table2text Encoder without field input
    """

    def __init__(self, uni_size, hidden_size):
        super(NonFieldLstmEncoder, self).__init__()
        self.uni_size = uni_size
        self.hidden_size = hidden_size
        self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size)

    def is_all_finished(self, finished):
        r""" Judge if Encode process has finished.

        Args:
            finished (Torch.Tensor): encoder finished tensor, shape: [batch_size].

        Returns:
            Torch.Tensor: flag of finished process, shape: [1].
        """
        flag = torch.BoolTensor(1).fill_(True).to(finished.device)
        for item in finished:
            flag &= item
        return flag

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of Lstm.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            tuple:
                - Torch.Tensor: the initial hidden states.
                - Torch.Tensor: the initial encoder finished tensor.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device

        hidden_states = (
            torch.zeros([batch_size, self.hidden_size],
                        dtype=torch.float32).to(device), torch.zeros([batch_size, self.hidden_size],
                                                                     dtype=torch.float32).to(device)
        )
        finished = torch.zeros([batch_size], dtype=torch.bool).to(device)
        return hidden_states, finished

    def forward(self, input_embeddings, input_length):
        r""" Implement the encoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            input_length (Torch.Tensor): length of input sequence, shape: [batch_size].

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, hidden_size].
                - tuple:
                    - Torch.Tensor: hidden state of direction 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden state of direction 2, shape: [batch_size, hidden_size].
        """
        hidden_states, finished = self.init_hidden(input_embeddings)

        input_ta = input_embeddings.permute(1, 0, 2)
        emit_ta = []

        t = 0
        while not self.is_all_finished(finished):
            o_t, hidden_states = self.enc_lstm(input_ta[t], hidden_states, finished)
            emit_ta.append(o_t)
            t += 1
            finished = t >= input_length

        outputs = torch.stack(emit_ta).to(input_embeddings.device)
        outputs = outputs.permute(1, 0, 2)

        return outputs, hidden_states


class FieldLstmEncoder(torch.nn.Module):
    r"""
    Lstm Table2text Encoder with field input
    """

    def __init__(self, uni_size, hidden_size, field_size):
        super(FieldLstmEncoder, self).__init__()
        self.uni_size = uni_size
        self.hidden_size = hidden_size
        self.field_size = field_size
        self.enc_lstm = FieldGateLstmUnit(self.hidden_size, self.uni_size, self.field_size)

    def is_all_finished(self, finished):
        r""" Judge if Encode process has finished.

        Args:
            finished (Torch.Tensor): encoder finished tensor, shape: [batch_size].

        Returns:
            Torch.Tensor: flag of finished process, shape: [1].
        """
        flag = torch.BoolTensor(1).fill_(True).to(finished.device)
        for item in finished:
            flag &= item
        return flag

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of Lstm.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            tuple:
                - Torch.Tensor: the initial hidden states.
                - Torch.Tensor: the initial encoder finished tensor.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device

        hidden_states = (
            torch.zeros([batch_size, self.hidden_size],
                        dtype=torch.float32).to(device), torch.zeros([batch_size, self.hidden_size],
                                                                     dtype=torch.float32).to(device)
        )
        finished = torch.zeros([batch_size], dtype=torch.bool).to(device)
        return hidden_states, finished

    def forward(self, input_embeddings, field_embeddings, input_length):
        r""" Implement the encoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            field_embeddings (Torch.Tensor): source field sequence embedding, shape: [batch_size, sequence_length, field_embedding_size].
            input_length (Torch.Tensor): length of input sequence, shape: [batch_size].

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, hidden_size].
                - tuple:
                    - Torch.Tensor: hidden state of direction 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden state of direction 2, shape: [batch_size, hidden_size].
        """
        hidden_states, finished = self.init_hidden(input_embeddings)

        input_ta = input_embeddings.permute(1, 0, 2)
        field_ta = field_embeddings.permute(1, 0, 2)
        emit_ta = []

        t = 0
        while not self.is_all_finished(finished):
            o_t, hidden_states = self.enc_lstm(input_ta[t], field_ta[t], hidden_states, finished)
            emit_ta.append(o_t)
            t += 1
            finished = t >= input_length

        emit_ta = torch.stack(emit_ta).to(input_embeddings.device)
        emit_ta = emit_ta.permute(1, 0, 2)

        return emit_ta, hidden_states
