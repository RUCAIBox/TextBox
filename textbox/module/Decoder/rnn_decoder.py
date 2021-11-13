# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/26
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn

# UPDATE:
# @Time   : 2021/9/8
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com

r"""
RNN Decoder
###############
"""

import torch
from torch import nn
import torch.nn.functional as F
from textbox.module.Attention.attention_mechanism import LuongAttention, BahdanauAttention, MonotonicAttention, FieldDualAttentionWrapper, FieldAttentionWrapper
from textbox.module.layers import LstmUnit, OutputUnit


class BasicRNNDecoder(torch.nn.Module):
    r"""
    Basic Recurrent Neural Network (RNN) decoder.
    """

    def __init__(self, embedding_size, hidden_size, num_dec_layers, rnn_type, dropout_ratio=0.0):
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
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
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
        r""" Implement the decoding process.

        Args:
            input_embeddings (Torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [num_layers * num_directions, batch_size, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        if not isinstance(hidden_states, tuple):
            hidden_states = hidden_states.contiguous()
        outputs, hidden_states = self.decoder(input_embeddings, hidden_states)
        return outputs, hidden_states


class AttentionalRNNDecoder(torch.nn.Module):
    r"""
    Attention-based Recurrent Neural Network (RNN) decoder.
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        context_size,
        num_dec_layers,
        rnn_type,
        dropout_ratio=0.0,
        attention_type='LuongAttention',
        alignment_method='concat'
    ):
        super(AttentionalRNNDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_type = rnn_type
        self.attention_type = attention_type
        self.alignment_method = alignment_method

        if attention_type == 'LuongAttention':
            self.attentioner = LuongAttention(self.context_size, self.hidden_size, self.alignment_method)
            dec_input_size = embedding_size
        elif attention_type == 'BahdanauAttention':
            self.attentioner = BahdanauAttention(self.context_size, self.hidden_size)
            dec_input_size = embedding_size + context_size
        elif attention_type == 'MonotonicAttention':
            self.attentioner = MonotonicAttention(self.context_size, self.hidden_size)
            dec_input_size = embedding_size
        else:
            raise ValueError("Attention type must be in ['LuongAttention', 'BahdanauAttention', 'MonotonicAttention'].")

        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'gru':
            self.decoder = nn.GRU(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'rnn':
            self.decoder = nn.RNN(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("RNN type in attentional decoder must be in ['lstm', 'gru', 'rnn'].")

        self.attention_dense = nn.Linear(hidden_size + context_size, hidden_size)

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
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_type))

    def forward(
        self, input_embeddings, hidden_states=None, encoder_outputs=None, encoder_masks=None, previous_probs=None
    ):
        r""" Implement the attention-based decoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.
            encoder_outputs (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_masks (Torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        if encoder_outputs is not None and encoder_masks is None:
            encoder_masks = torch.ones(encoder_outputs.size(0), encoder_outputs.size(1)).to(encoder_outputs.device)

        decode_length = input_embeddings.size(1)

        probs = previous_probs
        all_outputs = []
        for step in range(decode_length):

            if self.attention_type == 'BahdanauAttention':
                # only top layer
                if self.rnn_type == 'lstm':
                    hidden = hidden_states[0][-1]
                else:
                    hidden = hidden_states[-1]
                context, probs = self.attentioner(hidden, encoder_outputs, encoder_masks)
                embed = input_embeddings[:, step, :].unsqueeze(1)
                inputs = torch.cat((embed, context), dim=-1)
            else:
                inputs = input_embeddings[:, step, :].unsqueeze(1)
                context = None

            if (not isinstance(hidden_states, tuple)):
                hidden_states = hidden_states.contiguous()
            outputs, hidden_states = self.decoder(inputs, hidden_states)

            if self.attention_type == 'LuongAttention' and context is None:
                context, probs = self.attentioner(outputs, encoder_outputs, encoder_masks)
            elif self.attention_type == 'MonotonicAttention' and context is None:
                if self.training:
                    context, probs = self.attentioner.soft(outputs, encoder_outputs, encoder_masks, probs)
                else:
                    context, probs = self.attentioner.hard(outputs, encoder_outputs, encoder_masks, probs)
            elif self.attention_type == 'BahdanauAttention':
                pass
            else:
                raise NotImplementedError("No such attention type {} for decoder output.".format(self.attention_type))
            outputs = self.attention_dense(torch.cat((outputs, context), dim=2))
            all_outputs.append(outputs)

        outputs = torch.cat(all_outputs, dim=1)
        return outputs, hidden_states, probs


class FieldAttentionalRNNDecoder(torch.nn.Module):
    r"""
    Attention-based Recurrent Neural Network (RNN) decoder with Field context.
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        target_vocab_size,
        field_size,
        sos_token_idx,
        eos_token_idx,
        max_length,
        attention_type='FieldDualAttentionWrapper',
    ):
        super(FieldAttentionalRNNDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.target_vocab_size = target_vocab_size
        self.field_size = field_size
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.max_length = max_length
        self.attention_type = attention_type

        if attention_type == 'FieldDualAttentionWrapper':
            self.attentioner = FieldDualAttentionWrapper(self.hidden_size, self.hidden_size, self.field_size)
        elif attention_type == 'FieldAttentionWrapper':
            self.attentioner = FieldAttentionWrapper(self.hidden_size, self.hidden_size)
        else:
            raise ValueError("Attention type must be in ['FieldDualAttentionWrapper', 'FieldAttentionWrapper'].")

        self.decoder_lstm = LstmUnit(self.hidden_size, self.embedding_size)
        self.dec_out = OutputUnit(self.hidden_size, self.target_vocab_size)

    def is_all_finished(self, finished):
        r""" Judge if Decode process has finished.

        Args:
            finished (Torch.Tensor): decoder finished tensor, shape: [batch_size].

        Returns:
            Torch.Tensor: flag of finished process, shape: [1].
        """
        flag = torch.BoolTensor(1).fill_(True).to(finished.device)
        for item in finished:
            flag &= item
        return flag

    def set_attention_hiddenstate(self, encoder_output, field_embedding):
        '''

        Args:
            encoder_output (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size].
            field_embedding (Torch.Tensor): field embedding, shape: [batch_size, sequence_length, hidden_size]

        Returns:

        '''
        self.attentioner.set_hidden_state(encoder_output, field_embedding)

    def forward(self, hidden_states, input_embeddings, inputs_len, embedding):
        r""" Implement the attention-based decoding process.

        Args:
            hidden_states (tuple): initial hidden states of both directions
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            inputs_len (Torch.Tensor): length of input sequence, shape: [batch_size].
            embedding (nn.Embedding): Embedding unit, (vocab_size, embedding_size)

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, vocab_size].
                - tuple:
                    - Torch.Tensor: hidden state of direction 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden state of direction 2, shape: [batch_size, hidden_size].
        """
        batch_size = input_embeddings.size(0)

        h0 = hidden_states
        finished = torch.zeros([batch_size], dtype=torch.bool).to(input_embeddings.device)
        x0 = embedding(torch.zeros((batch_size)).long().fill_(self.sos_token_idx).to(input_embeddings.device))
        input_embeddings_ta = input_embeddings.permute(1, 0, 2)
        emit_ta = []

        o_t, h = self.decoder_lstm(x0, h0, finished)
        o_t, _ = self.attentioner(o_t)
        o_t = self.dec_out(o_t, finished)
        emit_ta.append(o_t)
        t = 0
        while not self.is_all_finished(finished):
            o_t, h = self.decoder_lstm(input_embeddings_ta[t], h, finished)
            o_t, _ = self.attentioner(o_t)
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            t += 1
            finished = t >= inputs_len

        emit_ta = torch.stack(emit_ta).to(input_embeddings.device)
        emit_ta = emit_ta.permute(1, 0, 2)

        return emit_ta, h

    def forward_test(self, initial_state, embedding):
        r""" Implement the attention-based decoding generate process.

        Args:
            initial_state (tuple): initial hidden states of both directions
            embedding (nn.Embedding): Embedding unit, (vocab_size, embedding_size)

        Returns:
            tuple:
                - Torch.Tensor: sequence of predicted token idx
                - Torch.Tensor: attention score
        """
        batch_size = initial_state[0].size(0)

        h0 = initial_state
        finished = torch.zeros([batch_size], dtype=torch.bool).to(initial_state[0].device)
        x0 = embedding(torch.zeros([batch_size]).long().fill_(self.sos_token_idx).to(initial_state[0].device))
        emit_ta = []
        att_ta = []

        o_t, h = self.decoder_lstm(x0, h0, finished)
        o_t, w_t = self.attentioner(o_t)
        o_t = self.dec_out(o_t, finished)
        emit_ta.append(o_t)
        att_ta.append(w_t)
        next_token = torch.argmax(o_t, 1)
        x = embedding(next_token)
        finished = torch.logical_or(finished, next_token == self.eos_token_idx)

        t = 0
        while not self.is_all_finished(finished):
            o_t, h = self.decoder_lstm(x, h, finished)
            o_t, w_t = self.attentioner(o_t)
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            att_ta.append(w_t)
            next_token = torch.argmax(o_t, 1)
            x = embedding(next_token)
            finished = torch.logical_or(finished, next_token == self.eos_token_idx)
            finished = torch.logical_or(finished, torch.BoolTensor([t >= self.max_length]).to(finished.device))
            t += 1

        outputs = torch.stack(emit_ta).to(initial_state[0].device)
        outputs = outputs.permute(1, 0, 2)
        pred_tokens = torch.argmax(outputs, 2)
        atts = torch.stack(att_ta).to(initial_state[0].device)

        return pred_tokens, atts
