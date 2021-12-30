# @Time   : 2020/12/28
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/25
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

r"""
RNNEncDec
################################################
Reference:
    Sutskever et al. "Sequence to Sequence Learning with Neural Networks" in NIPS 2014.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import PointerRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import Copy_Beam_Search


class PointerNet(Seq2SeqGenerator):
    r"""RNN-based Encoder-Decoder architecture is a basic framework for Seq2Seq text generation.
    """

    def __init__(self, config, dataset):
        super(PointerNet, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.strategy = config['decoding_strategy']

        self.is_attention = config['is_attention']
        self.is_pgen = config['is_pgen'] and self.is_attention
        self.is_coverage = config['is_coverage'] and self.is_attention

        if self.is_coverage:
            self.cov_loss_lambda = config['cov_loss_lambda']

        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.context_size = self.hidden_size

        # define layers and loss
        self.source_token_embedder = nn.Embedding(
            self.source_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        if config['share_vocab']:
            self.target_token_embedder = self.source_token_embedder
        else:
            self.target_token_embedder = nn.Embedding(
                self.target_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
            )

        self.encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type, self.dropout_ratio,
            self.bidirectional
        )

        self.decoder = PointerRNNDecoder(
            self.target_vocab_size, self.embedding_size, self.hidden_size, self.context_size,
            self.num_dec_layers, self.rnn_type, self.dropout_ratio,
            is_attention=self.is_attention, is_pgen=self.is_pgen, is_coverage=self.is_coverage
        )

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, batch_data, eval_data):
        generated_corpus = []

        source_text = batch_data['source_idx']
        source_length = batch_data['source_length']
        source_embeddings = self.source_token_embedder(source_text)
        encoder_outputs, encoder_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.rnn_type == 'lstm'):
                encoder_states = (encoder_states[0][::2], encoder_states[1][::2])
            else:
                encoder_states = encoder_states[::2]

        for bid in range(source_text.size(0)):
            # assert self.rnn_type = 'lstm'
            decoder_states = (encoder_states[0][:, bid, :].unsqueeze(1),
                              encoder_states[1][:, bid, :].unsqueeze(1))
            generated_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

            kwargs = {}
            if self.is_attention:
                kwargs['encoder_outputs'] = encoder_outputs[bid, :, :].unsqueeze(0)
                kwargs['encoder_masks'] = torch.ne(source_text[bid], self.padding_token_idx).unsqueeze(0).to(self.device)
                kwargs['context'] = torch.zeros((1, 1, self.context_size)).to(self.device)

            if self.is_pgen:
                kwargs['extra_zeros'] = batch_data['extra_zeros'][bid, :].unsqueeze(0)
                kwargs['source_extended_idx'] = batch_data['source_extended_idx'][bid, :].unsqueeze(0)
                kwargs['source_oovs'] = batch_data['source_oovs'][bid]

            if self.is_coverage:
                kwargs['coverages'] = torch.zeros((1, 1, source_text.size(1))).to(self.device)

            if self.strategy == 'beam_search':
                hypothesis = Copy_Beam_Search(
                    self.beam_size, self.sos_token_idx, self.eos_token_idx, self.unknown_token_idx,
                    self.device, self.idx2token,
                    is_attention=self.is_attention, is_pgen=self.is_pgen, is_coverage=self.is_coverage
                )

            for gen_id in range(self.target_max_length):
                input_embeddings = self.target_token_embedder(input_seq)

                vocab_dists, decoder_states, kwargs = self.decoder(
                    input_embeddings, decoder_states, kwargs=kwargs
                )

                if self.strategy == 'greedy_search':
                    token_idx = vocab_dists.view(-1).argmax().item()
                    if token_idx == self.eos_token_idx:
                        break
                    else:
                        if token_idx >= self.max_vocab_size:
                            generated_tokens.append(kwargs['oovs'][token_idx - self.max_vocab_size])
                            token_idx = self.unknown_token_idx
                        else:
                            generated_tokens.append(self.idx2token[token_idx])
                        input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                elif self.strategy == 'beam_search':
                    input_seq, decoder_states, kwargs = hypothesis.step(
                        gen_id, vocab_dists, decoder_states, kwargs)
                    if hypothesis.stop():
                        break

            if self.strategy == 'beam_search':
                generated_tokens = hypothesis.generate()

            generated_corpus.append(generated_tokens)

        return generated_corpus

    def forward(self, corpus, epoch_idx=0):
        source_text = corpus['source_idx']
        source_length = corpus['source_length']
        input_text = corpus['target_input_idx']

        source_embeddings = self.source_token_embedder(source_text)  # B x src_len x 128
        input_embeddings = self.target_token_embedder(input_text)  # B x dec_len x 128
        encoder_outputs, encoder_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.rnn_type == 'lstm'):
                encoder_states = (encoder_states[0][::2], encoder_states[1][::2])
            else:
                encoder_states = encoder_states[::2]

        batch_size = len(source_text)
        src_len = len(source_text[0])

        kwargs = {}
        if self.is_attention:
            kwargs['encoder_outputs'] = encoder_outputs  # B x src_len x 256
            kwargs['encoder_masks'] = torch.ne(source_text, self.padding_token_idx).to(self.device)  # B x src_len
            kwargs['context'] = torch.zeros((batch_size, 1, self.context_size)).to(self.device)  # B x 1 x 256

        if self.is_pgen:
            kwargs['extra_zeros'] = corpus['extra_zeros']  # B x max_oovs_num
            kwargs['source_extended_idx'] = corpus['source_extended_idx']  # B x src_len

        if self.is_coverage:
            kwargs['coverages'] = torch.zeros((batch_size, 1, src_len)).to(self.device)  # B x 1 x src_len

        vocab_dists, _, kwargs = self.decoder(
            input_embeddings, encoder_states, kwargs=kwargs
        )  # B x tgt_len x vocab_size+(max_oovs_num)

        target_output_idx = corpus['target_output_idx']  # B x tgt_len
        probs_masks = torch.ne(target_output_idx, self.padding_token_idx)  # mask pad idx probs
        gold_probs = torch.gather(vocab_dists, 2, target_output_idx.unsqueeze(2)).squeeze(2)  # B x tgt_len
        nll_loss = -torch.log(gold_probs + 1e-12)
        if self.is_coverage:
            coverage_loss = torch.sum(torch.min(kwargs['attn_dists'], kwargs['coverages']), dim=2)  # B x tgt_len
            nll_loss = nll_loss + self.cov_loss_lambda * coverage_loss
        loss = nll_loss * probs_masks
        length = corpus['target_length']
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss
