# @Time   : 2021/8/13
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com

import torch
from torch import nn
import torch.nn.functional as F
from textbox.model.abstract_generator import Seq2SeqGenerator


class FieldGateLstmUnit(nn.Module):
    r"""FieldGateLstmUnit

        Args:
            - hidden_size(int): size of hidden state.
            - input_size(int): size of input
            - field_size(int): size of field input, default: None
    """

    def __init__(self, hidden_size, input_size, field_size=None):
        super(FieldGateLstmUnit, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.linear_1 = nn.Linear(self.input_size + self.hidden_size, 4 * self.hidden_size)
        if field_size is not None:
            self.field_size = field_size
            self.linear_2 = nn.Linear(self.field_size, 2 * self.hidden_size)

    def forward(self, input, state, finished=None, field_input=None):
        r""" Implement the encoding process.

        Args:
            input (Torch.Tensor): source embedding, shape: [batch_size, embedding_size].
            state (tuple): state information, shape: [batch_size, hidden_size].
            finished (Torch.Tensor): finished flag tensor, default: None.
            field_input (Torch.Tensor): field embedding, shape: [batch_size, field_size], default:None.

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

        # Final Memory cell
        if field_input is not None:
            field_input = self.linear_2(field_input)
            r, d = torch.split(field_input, field_input.shape[1] // 2, 1)
            c = torch.sigmoid(f + 1.0) * c_prev + torch.sigmoid(i) * torch.tanh(j) + torch.sigmoid(r) * torch.tanh(d)
        else:
            c = torch.sigmoid(f + 1.0) * c_prev + torch.sigmoid(i) * torch.tanh(j)
        h = torch.sigmoid(o) * torch.tanh(c)

        out, state = h, (h, c)

        if finished is not None:
            tmp_finished = finished.unsqueeze(-1).expand(-1, h.shape[1])
            out = torch.where(tmp_finished, torch.zeros_like(h).to(finished.device), h)
            state = (torch.where(tmp_finished, h_prev, h), torch.where(tmp_finished, c_prev, c))

        return out, state


class FieldDualAttentionWrapper(torch.nn.Module):

    def __init__(self, hidden_size, input_size, field_size=None):
        super(FieldDualAttentionWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(input_size, hidden_size)
        self.linear_3 = nn.Linear(2 * input_size, hidden_size)
        if self.field_size is not None:
            self.linear_4 = nn.Linear(field_size, hidden_size)
            self.linear_5 = nn.Linear(input_size, hidden_size)

    def set_hidden_state(self, hidden_state, field_hidden_state=None):
        self.hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state_2d = self.hidden_state.reshape([-1, self.input_size])
        phi_hidden_state_2d = torch.tanh(self.linear_1(hidden_state_2d))
        self.phi_hidden_state = phi_hidden_state_2d.reshape(self.hidden_state.shape)

        if field_hidden_state is not None:
            self.field_hidden_state = field_hidden_state.permute(1, 0, 2)
            field_hidden_state_2d = self.field_hidden_state.reshape([-1, self.field_size])
            phi_field_state_2d = torch.tanh(self.linear_4(field_hidden_state_2d))
            self.phi_field_state = phi_field_state_2d.reshape(self.hidden_state.shape)

    def forward(self, input, finished=None):
        gamma_h = torch.tanh(self.linear_2(input))
        weights = torch.sum(self.phi_hidden_state * gamma_h, dim=2, keepdim=True)
        weights = torch.exp(weights - torch.max(weights, dim=0, keepdim=True)[0])
        weights = torch.divide(weights, (1e-6 + torch.sum(weights, dim=0, keepdim=True)))
        if self.field_size is not None:
            alpha_h = torch.tanh(self.linear_5(input))
            field_weights = torch.sum(self.phi_field_state * alpha_h, dim=2, keepdim=True)
            field_weights = torch.exp(field_weights - torch.max(field_weights, dim=0, keepdim=True)[0])
            field_weights = torch.divide(field_weights, (1e-6 + torch.sum(field_weights, dim=0, keepdim=True)))
            weights = torch.divide(
                weights * field_weights, (1e-6 + torch.sum(weights * field_weights, dim=0, keepdim=True))
            )

        context = torch.sum(self.hidden_state * weights, dim=0)
        out = self.linear_3(torch.cat([context, input], -1))

        if finished is not None:
            out = torch.where(finished, torch.zeros_like(out), out)
        return out, weights


def is_all_finished(finished):
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

    def forward(self, input_embeddings, input_length, field_embeddings=None):
        r""" Implement the encoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            input_length (Torch.Tensor): length of input sequence, shape: [batch_size].
            field_embeddings (Torch.Tensor): source field sequence embedding, shape: [batch_size, sequence_length, field_embedding_size], default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, hidden_size].
                - tuple:
                    - Torch.Tensor: hidden 1, shape: [batch_size, hidden_size].
                    - Torch.Tensor: hidden 2, shape: [batch_size, hidden_size].
        """
        hidden_states, finished = self.init_hidden(input_embeddings)

        input_ta = input_embeddings.permute(1, 0, 2)
        if self.field_size is not None:
            field_ta = field_embeddings.permute(1, 0, 2)
        else:
            field_ta = [None] * input_length
        emit_ta = []
        t = 0
        while not is_all_finished(finished):
            o_t, hidden_states = self.enc_lstm(input_ta[t], hidden_states, finished, field_ta[t])
            emit_ta.append(o_t)
            t += 1
            finished = t >= input_length

        emit_ta = torch.stack(emit_ta).to(input_embeddings.device)
        emit_ta = emit_ta.permute(1, 0, 2)

        return emit_ta, hidden_states


class FieldAttentionalRNNDecoder(torch.nn.Module):
    r"""
    Attention-based Recurrent Neural Network (RNN) decoder with Field context.
    """

    def __init__(
        self, embedding_size, hidden_size, target_vocab_size, field_size, sos_token_idx, eos_token_idx, max_length,
        dual_att
    ):
        super(FieldAttentionalRNNDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.target_vocab_size = target_vocab_size
        self.field_size = field_size if dual_att else None
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.max_length = max_length
        self.attentioner = FieldDualAttentionWrapper(self.hidden_size, self.hidden_size, self.field_size)

        self.decoder_lstm = FieldGateLstmUnit(self.hidden_size, self.embedding_size)
        self.dec_out = nn.Linear(self.hidden_size, self.target_vocab_size)

    def forward(self, initial_state, embedding, input_embeddings=None, inputs_len=None):
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
        if input_embeddings is not None:
            input_embeddings_ta = input_embeddings.permute(1, 0, 2)

        o_t, h = self.decoder_lstm(x0, h0, finished)
        o_t, w_t = self.attentioner(o_t)
        o_t = self.dec_out(o_t)
        emit_ta.append(o_t)
        att_ta.append(w_t)
        t = 0
        while not is_all_finished(finished):
            x = embedding(torch.argmax(o_t, 1)) if input_embeddings is None else input_embeddings_ta[t]
            o_t, h = self.decoder_lstm(x, h, finished)
            o_t, w_t = self.attentioner(o_t)
            o_t = self.dec_out(o_t)
            emit_ta.append(o_t)
            att_ta.append(w_t)
            next_token = torch.argmax(o_t, 1)
            finished = torch.logical_or(finished, next_token == self.eos_token_idx)
            finished = torch.logical_or(finished, torch.BoolTensor([t >= self.max_length]).to(finished.device))
            t += 1
            if inputs_len is not None:
                finished = t >= inputs_len

        outputs = torch.stack(emit_ta).to(initial_state[0].device)
        outputs = outputs.permute(1, 0, 2)
        atts = torch.stack(att_ta).to(initial_state[0].device)

        return outputs, atts, h


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
        self.fgate_enc = config['fgate_encoder']
        self.field_encoder_size = self.field_size if not self.encoder_add_pos else self.field_size + 2 * self.pos_size
        self.field_encoder_size = self.field_encoder_size if self.fgate_enc else None
        self.field_attention_size = self.field_size if not self.decoder_add_pos else self.field_size + 2 * self.pos_size
        self.position_vocab = config['position_vocab']
        self.max_length = config['max_length']
        self.dual_att = config['dual_attention']

        # ======================================== encoder =========================================== #
        self.encoder = FieldLstmEncoder(self.uni_size, self.hidden_size, self.field_encoder_size)

        # ======================================== embeddings ======================================== #
        self.embedding = nn.Embedding(self.source_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.fembedding = nn.Embedding(self.source_key_vocab_size, self.field_size)

        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pembedding = nn.Embedding(self.position_vocab, self.pos_size)
            self.rembedding = nn.Embedding(self.position_vocab, self.pos_size)

        # ======================================== decoder =========================================== #
        self.decoder = FieldAttentionalRNNDecoder(
            self.embedding_size, self.hidden_size, self.target_vocab_size, self.field_attention_size,
            self.sos_token_idx, self.eos_token_idx, self.max_length, self.dual_att
        )

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

            text.append(_text)
            field.append(_field)
            pos.append(_pos)
            rpos.append(_rpos)

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

        batch_data['enc_in'] = torch.tensor(batch_data['enc_in']).to(self.device)
        batch_data['enc_len'] = torch.IntTensor(batch_data['enc_len']).to(self.device)
        batch_data['enc_fd'] = torch.tensor(batch_data['enc_fd']).to(self.device)
        batch_data['enc_pos'] = torch.tensor(batch_data['enc_pos']).to(self.device)
        batch_data['enc_rpos'] = torch.tensor(batch_data['enc_rpos']).to(self.device)
        batch_data['dec_in'] = torch.tensor(batch_data['dec_in']).to(self.device)
        batch_data['dec_len'] = torch.IntTensor(batch_data['dec_len']).to(self.device)
        batch_data['dec_out'] = torch.tensor(batch_data['dec_out']).to(self.device)

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

        encoder_embedding = self.embedding(batch_data['enc_in'])
        decoder_embedding = self.embedding(batch_data['dec_in'])
        field_pos_embedding = None
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            field_embedding = self.fembedding(batch_data['enc_fd'])
            field_pos_embedding = field_embedding
            if self.field_concat:
                encoder_embedding = torch.cat([encoder_embedding, field_embedding], 2)

        if (self.position_concat or self.encoder_add_pos or self.decoder_add_pos) and field_embedding is not None:
            pos_embedding = self.pembedding(batch_data['enc_pos'])
            rpos_embedding = self.rembedding(batch_data['enc_rpos'])
            if self.position_concat:
                encoder_embedding = torch.cat([encoder_embedding, pos_embedding, rpos_embedding], 2)
                field_pos_embedding = torch.cat([field_embedding, pos_embedding, rpos_embedding], 2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                field_pos_embedding = torch.cat([field_embedding, pos_embedding, rpos_embedding], 2)

        # ===================================== encode process ======================================= #
        en_output, en_state = self.encoder(
            encoder_embedding,
            batch_data['enc_len'],
            field_pos_embedding,
        )

        self.decoder.attentioner.set_hidden_state(en_output, field_pos_embedding)

        return batch_data, en_state, decoder_embedding

    def forward(self, batch_data, epoch_idx=0):
        r""" Implement the forward process.

        Args:
            batch_data (dict): batch data from data loader

        Returns:
            Torch.Tensor: the loss of network, shape: [1]
        """
        batch_data, en_state, decoder_embedding = self.prepare(batch_data)
        de_output, _, de_state = self.decoder(en_state, self.embedding, decoder_embedding, batch_data['dec_len'])
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
        de_output, atts, _ = self.decoder(en_state, self.embedding)
        predictions = torch.argmax(de_output, 2)

        pred_list = []
        idx = 0
        for i, summary in enumerate(predictions):
            real_sum = []
            for tk, tid in enumerate(summary):
                tid = tid.item()
                if tid == self.unknown_token_idx:
                    sub = texts[i][torch.argmax(atts[tk, :len(texts[i]), idx])]
                    real_sum.append(sub)
                elif tid == self.eos_token_idx:
                    break
                else:
                    real_sum.append(idx2token[tid])

            pred_list.append([str(x) for x in real_sum])
            idx += 1

        return pred_list
