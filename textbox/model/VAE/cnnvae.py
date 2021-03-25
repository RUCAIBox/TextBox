# @Time   : 2020/11/9
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

r"""
CNNVAE
################################################
Reference:
    Yang et al. "Improved Variational Autoencoders for Text Modeling using Dilated Convolutions" in ICML 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from textbox.model.abstract_generator import UnconditionalGenerator
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.cnn_decoder import BasicCNNDecoder
from textbox.module.layers import Highway
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling


class CNNVAE(UnconditionalGenerator):
    r""" CNNVAE use a dilated CNN as decoder, which made a trade-off between contextual capacity of the decoder and
    effective use of encoding information.
    """

    def __init__(self, config, dataset):
        super(CNNVAE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_highway_layers = config['num_highway_layers']
        self.rnn_type = config['rnn_type']
        self.max_epoch = config['epochs']
        self.decoder_kernel_size = config['decoder_kernel_size']
        self.decoder_dilations = config['decoder_dilations']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.eval_generate_num = config['eval_generate_num']
        self.max_length = config['max_seq_length']

        self.num_directions = 2 if self.bidirectional else 1
        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        self.encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type, self.dropout_ratio,
            self.bidirectional
        )
        self.decoder = BasicCNNDecoder(
            self.embedding_size, self.latent_size, self.decoder_kernel_size, self.decoder_dilations, self.dropout_ratio
        )
        self.highway_1 = Highway(self.num_highway_layers, self.embedding_size)
        self.highway_2 = Highway(self.num_highway_layers, self.num_directions * self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.decoder_kernel_size[-1], self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        self.hidden_to_mean = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
        self.hidden_to_logvar = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, batch_data, eval_data):
        generate_corpus = []
        idx2token = eval_data.idx2token
        z = torch.randn(size=(1, self.latent_size), device=self.device)
        generate_tokens = []
        input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
        for _ in range(self.max_length):
            decoder_input = self.token_embedder(input_seq)
            outputs = self.decoder(decoder_input=decoder_input, noise=z)
            token_logits = self.vocab_linear(outputs)
            token_idx = topk_sampling(token_logits)
            token_idx = token_idx.item()
            if token_idx == self.eos_token_idx:
                break
            else:
                generate_tokens.append(idx2token[token_idx])
                input_seq = torch.LongTensor([[token_idx]]).to(self.device)
        generate_corpus.append(generate_tokens)
        return generate_corpus

    def forward(self, corpus, epoch_idx=0):
        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]
        input_length = corpus['target_length'] - 1
        batch_size = input_text.size(0)

        input_emb = self.token_embedder(input_text)
        input_emb = self.highway_1(input_emb)

        _, hidden_states = self.encoder(input_emb, input_length)

        if self.rnn_type == "lstm":
            h_n, c_n = hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            h_n = hidden_states
        else:
            raise NotImplementedError("No such rnn type {} for CNNVAE.".format(self.rnn_type))

        if self.bidirectional:
            h_n = h_n.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]

        h_n = self.highway_2(h_n)

        mean = self.hidden_to_mean(h_n)
        logvar = self.hidden_to_logvar(h_n)

        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = mean + z * torch.exp(0.5 * logvar)

        outputs = self.decoder(decoder_input=input_emb, noise=z)
        token_logits = self.vocab_linear(outputs)

        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        length = corpus['target_length'] - 1
        loss = loss.sum(dim=1) / length.float()

        kld_coef = float(epoch_idx / self.max_epoch) + 1e-3
        kld = -0.5 * torch.sum(logvar - mean.pow(2) - logvar.exp() + 1, 1).mean()

        # gradually increase the kld weight
        loss = loss.mean() + kld_coef * kld
        return loss

    def calculate_nll_test(self, corpus, epoch_idx=0):
        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]
        input_length = corpus['target_length'] - 1
        batch_size = input_text.size(0)

        input_emb = self.token_embedder(input_text)
        input_emb = self.highway_1(input_emb)

        _, hidden_states = self.encoder(input_emb, input_length)

        if self.rnn_type == "lstm":
            h_n, c_n = hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            h_n = hidden_states
        else:
            raise NotImplementedError("No such rnn type {} for CNNVAE.".format(self.rnn_type))

        if self.bidirectional:
            h_n = h_n.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]

        h_n = self.highway_2(h_n)

        mean = self.hidden_to_mean(h_n)
        logvar = self.hidden_to_logvar(h_n)

        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = mean + z * torch.exp(0.5 * logvar)

        outputs = self.decoder(decoder_input=input_emb, noise=z)
        token_logits = self.vocab_linear(outputs)

        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        loss = loss.sum(dim=1)
        return loss.mean()
