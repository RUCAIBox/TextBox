# @Time   : 2021/8/13
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from textbox.model.abstract_generator import Seq2SeqGenerator


class FieldGateLstmUnit(nn.Module):
    r"""FieldGateLstmUnit

        Args:
            - hidden_size(int): size of hidden state.
            - input_size(int): size of input
            - field_size(int): size of field input
    """

    def __init__(self, hidden_size, input_size, field_size):
        super(FieldGateLstmUnit, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size

        self.linear_1 = nn.Linear(self.input_size + self.hidden_size, 4 * self.hidden_size)
        self.linear_2 = nn.Linear(self.field_size, 2 * self.hidden_size)

    def forward(self, input, field_input, state, finished=None):
        r""" Implement the encoding process.

        Args:
            input (Torch.Tensor): source embedding, shape: [batch_size, embedding_size].
            field_input (Torch.Tensor): field embedding, shape: [batch_size, field_size]
            state (tuple): state information, shape: [batch_size, hidden_size].
            finished (Torch.Tensor): finished flag tensor, default: None

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, hidden_size].
                - tuple:
                    - Torch.Tensor: hidden state 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden state 2, shape: [batch_size, hidden_size].
        """
        h_prev, c_prev = state

        input = torch.cat([input, h_prev], 1)
        input = self.linear_1(input)
        i, j, f, o = torch.split(input, input.shape[1] // 4, 1)

        field_input = self.linear_2(field_input)

        r, d = torch.split(field_input, field_input.shape[1] // 2, 1)

        # Final Memory cell
        c = torch.sigmoid(f + 1.0) * c_prev + torch.sigmoid(i) * torch.tanh(j) + torch.sigmoid(r) * torch.tanh(d)
        h = torch.sigmoid(o) * torch.tanh(c)

        out, state = h, (h, c)

        if finished is not None:
            tmp_finished = finished.unsqueeze(-1).expand(-1, h.shape[1])
            out = torch.where(tmp_finished, torch.zeros_like(h).to(finished.device), h)
            state = (torch.where(tmp_finished, h_prev, h), torch.where(tmp_finished, c_prev, c))

        return out, state


class LstmUnit(nn.Module):
    r"""LstmUnit

        Args:
            - hidden_size(int): size of hidden state.
            - input_size(int): size of input
    """

    def __init__(self, hidden_size, input_size):
        super(LstmUnit, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.fc = nn.Linear(self.input_size + self.hidden_size, 4 * self.hidden_size)

    def forward(self, input, state, finished=None):
        r""" Implement the encoding process.

        Args:
            input (Torch.Tensor): source embedding, shape: [batch_size, embedding_size].
            state (tuple): state information, shape: [batch_size, hidden_size].
            finished (Torch.Tensor): finished flag tensor, default: None

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, hidden_size].
                - tuple:
                    - Torch.Tensor: hidden state 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden state 2, shape: [batch_size, hidden_size].
        """
        h_prev, c_prev = state

        input = torch.cat([input, h_prev], 1)

        input = self.fc(input)
        i, j, f, o = torch.split(input, input.shape[1] // 4, 1)

        # Final Memory cell
        c = torch.sigmoid(f + 1.0) * c_prev + torch.sigmoid(i) * torch.tanh(j)
        h = torch.sigmoid(o) * torch.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            tmp_finished = finished.unsqueeze(-1).expand(-1, h.shape[1])
            out = torch.where(tmp_finished, torch.zeros_like(h).to(finished.device), h)
            state = (torch.where(tmp_finished, h_prev, h), torch.where(tmp_finished, c_prev, c))

        return out, state


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
                - tuple: the initial hidden states.
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
                    - Torch.Tensor: hidden state 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden state 2, shape: [batch_size, hidden_size].
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
                - tuple: the initial hidden states.
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
                    - Torch.Tensor: hidden 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden 2, shape: [batch_size, hidden_size].
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


class OutputUnit(nn.Module):
    r"""OutputUnit

        Args:
            - input_size(int): size of input.
            - output_size(int): size of Unit's output
    """

    def __init__(self, input_size, output_size):
        super(OutputUnit, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, finished=None):
        r""" Implement the linear output process.

        Args:
            x (Torch.Tensor): input embedding, shape: [batch_size, input_size].
            finished (Torch.Tensor): finished flag tensor, default: None

        Returns:
            - Torch.Tensor: linear output, shape: [batch_size, output_size].
        """
        out = self.linear(x)

        if finished is not None:
            tmp_finished = finished.unsqueeze(-1).expand(-1, out.shape[1])
            out = torch.where(tmp_finished, torch.zeros_like(out), out)
        return out


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


class Table2TextEncDec(Seq2SeqGenerator):
    r"""LSTM-based Encoder-Decoder architecture is a powerful framework for Seq2Seq table-to-text generation.
    """

    def __init__(self, config, dataset):
        super(Table2TextEncDec, self).__init__(config, dataset)
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['embedding_size']
        self.field_size = config['field_size']
        self.pos_size = config['pos_size']
        self.field_concat = config['field_concat']
        self.position_concat = config['position_concat']
        self.encoder_add_pos = config['encoder_add_pos']
        self.decoder_add_pos = config['decoder_add_pos']
        self.uni_size = self.embedding_size if not self.field_concat else self.embedding_size + self.field_size
        self.uni_size = self.uni_size if not self.position_concat else self.uni_size + 2 * self.pos_size
        self.field_encoder_size = self.field_size if not self.encoder_add_pos else self.field_size + 2 * self.pos_size
        self.field_attention_size = self.field_size if not self.decoder_add_pos else self.field_size + 2 * self.pos_size
        self.position_vocab = config['position_vocab']
        self.max_length = config['max_length']
        self.fgate_enc = config['fgate_encoder']
        self.dual_att = config['dual_attention']

        # ======================================== encoder =========================================== #
        if self.fgate_enc:
            self.encoder = FieldLstmEncoder(self.uni_size, self.hidden_size, self.field_encoder_size)
            print('field-gated encoder LSTM')
        else:
            self.encoder = NonFieldLstmEncoder(self.uni_size, self.hidden_size)
            print('normal encoder LSTM')

        # ======================================== embeddings ======================================== #
        self.embedding = nn.Embedding(self.source_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.fembedding = nn.Embedding(self.source_key_vocab_size, self.field_size)

        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pembedding = nn.Embedding(self.position_vocab, self.pos_size)
            self.rembedding = nn.Embedding(self.position_vocab, self.pos_size)

        # ======================================== decoder =========================================== #
        if self.dual_att:
            self.decoder = FieldAttentionalRNNDecoder(
                self.embedding_size,
                self.hidden_size,
                self.target_vocab_size,
                self.field_size,
                self.sos_token_idx,
                self.eos_token_idx,
                self.max_length,
                attention_type='FieldDualAttentionWrapper'
            )
            print('dual attention mechanism used')
        else:
            self.decoder = FieldAttentionalRNNDecoder(
                self.embedding_size,
                self.hidden_size,
                self.target_vocab_size,
                self.field_size,
                self.sos_token_idx,
                self.eos_token_idx,
                self.max_length,
                attention_type='FieldAttentionWrapper'
            )
            print("normal attention used")

    def data_preprocess(self, batch_data, _max_text_len=100):
        r""" Implement the data preparation process.

        Args:
            batch_data (dict): batch data from data loader
            _max_text_len (int): pre-set max len of source text length, default: 100

        Returns:
            dict: new form of batch data
        """
        key = batch_data['source_key_idx']
        value = batch_data['source_value_idx']
        summary = batch_data['target_idx']
        summary_len = batch_data['target_length']

        batch_size = len(key)

        text, field, pos, rpos = [], [], [], []

        for i in range(batch_size):
            _text, _field, _pos, _rpos = [], [], [], []
            for j in range(len(value[i])):
                size = len(value[i][j])
                _text.extend(value[i][j])
                _field.extend([key[i][j]] * size)
                if size < self.position_vocab:
                    _pos.extend(list(range(1, size + 1)))
                    _rpos.extend(list(range(1, size + 1))[::-1])
                else:
                    _pos.extend(list(range(1, self.position_vocab - 1)))
                    _pos.extend([30] * (size + 1 - (self.position_vocab - 1)))

                    _rpos.extend(list([30] * (size + 1 - (self.position_vocab - 1)))[::-1])
                    _rpos.extend(list(range(1, self.position_vocab - 1))[::-1])

            assert len(_text) == len(_pos)
            assert len(_text) == len(_rpos)
            text.append(_text)
            field.append(_field)
            pos.append(_pos)
            rpos.append(_rpos)

        if isinstance(summary, torch.Tensor):
            summary = summary.tolist()
        for i in range(batch_size):
            summary[i] = summary[i][1:summary_len[i] - 1]

        batch = list(zip(summary, text, field, pos, rpos))

        max_summary_len = max([len(item[0]) for item in batch])
        max_text_len = max([len(item[1]) for item in batch])

        batch_data = {
            'enc_in': [],
            'enc_fd': [],
            'enc_pos': [],
            'enc_rpos': [],
            'enc_len': [],
            'dec_in': [],
            'dec_len': [],
            'dec_out': []
        }

        for i in range(len(batch)):
            summary, text, field, pos, rpos = batch[i]
            summary_len = len(summary)
            text_len = len(text)
            pos_len = len(pos)
            rpos_len = len(rpos)
            assert text_len == len(field)
            assert pos_len == len(field)
            assert rpos_len == pos_len
            gold = summary + [self.eos_token_idx] + [self.padding_token_idx] * (max_summary_len - summary_len)
            summary = summary + [self.padding_token_idx] * (max_summary_len - summary_len)
            text = text + [self.padding_token_idx] * (max_text_len - text_len)
            field = field + [self.padding_token_idx] * (max_text_len - text_len)
            pos = pos + [self.padding_token_idx] * (max_text_len - text_len)
            rpos = rpos + [self.padding_token_idx] * (max_text_len - text_len)

            if max_text_len > _max_text_len:
                text = text[:_max_text_len]
                field = field[:_max_text_len]
                pos = pos[:_max_text_len]
                rpos = rpos[:_max_text_len]
                text_len = min(text_len, _max_text_len)

            batch_data['enc_in'].append(text)
            batch_data['enc_len'].append(text_len)
            batch_data['enc_fd'].append(field)
            batch_data['enc_pos'].append(pos)
            batch_data['enc_rpos'].append(rpos)
            batch_data['dec_in'].append(summary)
            batch_data['dec_len'].append(summary_len)
            batch_data['dec_out'].append(gold)

        batch_data['enc_in'] = torch.tensor(batch_data['enc_in'])
        batch_data['enc_len'] = torch.IntTensor(batch_data['enc_len'])
        batch_data['enc_fd'] = torch.tensor(batch_data['enc_fd'])
        batch_data['enc_pos'] = torch.tensor(batch_data['enc_pos'])
        batch_data['enc_rpos'] = torch.tensor(batch_data['enc_rpos'])
        batch_data['dec_in'] = torch.tensor(batch_data['dec_in'])
        batch_data['dec_len'] = torch.IntTensor(batch_data['dec_len'])
        batch_data['dec_out'] = torch.tensor(batch_data['dec_out'])

        return batch_data

    def prepare(self, batch_data):
        r""" Implement the general process of forward process and generate process.

        Args:
            batch_data (dict): batch data from data loader

        Returns:
            tuple:
                - dict: new form of batch data
                - tuple: stored states of encoder
                - Torch.tensor: embedding of decoder input
        """
        batch_data = self.data_preprocess(batch_data)
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(self.device)

        encoder_embedding = self.embedding(batch_data['enc_in'])
        decoder_embedding = self.embedding(batch_data['dec_in'])
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            field_embedding = self.fembedding(batch_data['enc_fd'])
            field_pos_embedding = field_embedding
            if self.field_concat:
                encoder_embedding = torch.cat([encoder_embedding, field_embedding], 2)

        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            pos_embedding = self.pembedding(batch_data['enc_pos'])
            rpos_embedding = self.rembedding(batch_data['enc_rpos'])
            if self.position_concat:
                encoder_embedding = torch.cat([encoder_embedding, pos_embedding, rpos_embedding], 2)
                field_pos_embedding = torch.cat([field_embedding, pos_embedding, rpos_embedding], 2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                field_pos_embedding = torch.cat([field_embedding, pos_embedding, rpos_embedding], 2)

        # ===================================== encode process ======================================= #
        if self.fgate_enc:
            # print('field gated encoder used')
            en_output, en_state = self.encoder(encoder_embedding, field_pos_embedding, batch_data['enc_len'])
        else:
            # print('normal encoder used')
            en_output, en_state = self.encoder(encoder_embedding, batch_data['enc_len'])
        self.decoder.set_attention_hiddenstate(en_output, field_pos_embedding)

        return batch_data, en_state, decoder_embedding

    def forward(self, batch_data, epoch_idx=0):
        r""" Implement the forward process.

        Args:
            batch_data (dict): batch data from data loader

        Returns:
            Torch.Tensor: the loss of network, shape: [1]
        """
        batch_data, en_state, decoder_embedding = self.prepare(batch_data)
        de_output, de_state = self.decoder(en_state, decoder_embedding, batch_data['dec_len'], self.embedding)
        target = batch_data['dec_out']
        loss = F.cross_entropy(de_output.permute(0, 2, 1), target, reduction='none')
        mask = torch.sign(target.float())
        loss = mask * loss
        loss = torch.mean(loss)
        return loss

    def generate(self, batch_data, eval_data):
        texts = batch_data['source_value_text']
        idx2token = eval_data.target_idx2token
        for i in range(len(texts)):
            tmp = []
            for item in texts[i]:
                tmp.extend(item)
            texts[i] = tmp

        # decoder for testing
        batch_data, en_state, decoder_embedding = self.prepare(batch_data)
        predictions, atts = self.decoder.forward_test(en_state, self.embedding)

        predictions = predictions.cpu().detach().numpy()
        atts = atts.cpu().detach().squeeze().numpy()

        pred_list = []
        idx = 0
        for i, summary in enumerate(np.array(predictions)):
            summary = list(summary)
            if self.eos_token_idx in summary:
                summary = summary[:summary.index(self.eos_token_idx)] if summary[0] != self.eos_token_idx else [
                    self.eos_token_idx
                ]
            real_sum, unk_sum, mask_sum = [], [], []
            for tk, tid in enumerate(summary):
                if tid == self.unknown_token_idx:
                    sub = texts[i][np.argmax(atts[tk, :len(texts[i]), idx])]
                    real_sum.append(sub)
                else:
                    real_sum.append(idx2token[tid])

            pred_list.append([str(x) for x in real_sum])
            idx += 1

        return pred_list
