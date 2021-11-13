import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from textbox.model.abstract_generator import Seq2SeqGenerator
from textbox.module.Encoder.rnn_encoder import FieldLstmEncoder, NonFieldLstmEncoder
from textbox.module.Decoder.rnn_decoder import FieldAttentionalRNNDecoder


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

        # ======================================== decoder ======================================== #
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

        # ======================================== encoder ======================================== #
        if self.fgate_enc:
            # print('field gated encoder used')
            en_output, en_state = self.encoder(encoder_embedding, field_pos_embedding, batch_data['enc_len'])
        else:
            # print('normal encoder used')
            en_output, en_state = self.encoder(encoder_embedding, batch_data['enc_len'])
        self.decoder.set_attention_hiddenstate(en_output, field_pos_embedding)

        return batch_data, en_state, decoder_embedding

    def forward(self, batch_data, epoch_idx=0):
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
