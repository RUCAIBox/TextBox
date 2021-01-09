# @Time   : 2020/11/9
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

r"""
HybridVAE
################################################
Reference:
    Rothe et al. "A Hybrid Convolutional Variational Autoencoder for Text Generation" in EMNLP 2017.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from textbox.model.abstract_generator import UnconditionalGenerator
from textbox.module.Encoder.cnn_encoder import BasicCNNEncoder
from textbox.module.Decoder.cnn_decoder import HybridDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling


class HybridVAE(UnconditionalGenerator):
    r"""HybridVAE blends fully feed-forward convolutional and deconvolutional components
     with a recurrent language model.
    """

    def __init__(self, config, dataset):
        super(HybridVAE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_dec_layers = config['num_dec_layers']
        self.max_epoch = config['epochs']
        self.alpha_aux = config['alpha_aux']
        self.rnn_type = config['rnn_type']
        self.dropout_ratio = config['dropout_ratio']
        self.eval_generate_num = config['eval_generate_num']
        self.max_length = config['max_seq_length']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        self.encoder = BasicCNNEncoder(self.embedding_size, self.latent_size)
        self.decoder = HybridDecoder(self.embedding_size, self.latent_size, self.hidden_size, self.num_dec_layers,
                                     self.rnn_type, self.vocab_size)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')
        self.hidden_to_mean = nn.Linear(self.latent_size, self.latent_size)
        self.hidden_to_logvar = nn.Linear(self.latent_size, self.latent_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, eval_data):
        generate_corpus = []
        idx2token = eval_data.idx2token

        with torch.no_grad():
            for _ in range(self.eval_generate_num):
                z = torch.randn(size=(1, self.latent_size), device=self.device)
                cnn_out = self.decoder.conv_decoder(z)
                if self.rnn_type == "lstm":
                    hidden_states = torch.randn(size=(1, 2 * self.hidden_size), device=self.device)
                    hidden_states = torch.chunk(hidden_states, 2, dim=-1)
                    h_0 = hidden_states[0].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                    c_0 = hidden_states[1].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                    hidden_states = (h_0, c_0)
                else:
                    hidden_states = torch.randn(size=(self.num_dec_layers, 1, self.hidden_size), device=self.device)
                generate_tokens = []
                input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
                for gen_idx in range(self.max_length):
                    decoder_input = self.token_embedder(input_seq)

                    token_logits, hidden_states = self.decoder.rnn_decoder(cnn_out[:, gen_idx, :].unsqueeze(1),
                                                                           decoder_input=decoder_input,
                                                                           initial_state=hidden_states)
                    token_idx = topk_sampling(token_logits)
                    token_idx = token_idx.item()
                    if token_idx == self.eos_token_idx:
                        break
                    else:
                        generate_tokens.append(idx2token[token_idx])
                        input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                generate_corpus.append(generate_tokens)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]
        batch_size = input_text.size(0)

        input_emb = self.token_embedder(input_text)
        hidden_states = self.encoder(input_emb)

        mean = self.hidden_to_mean(hidden_states)
        logvar = self.hidden_to_logvar(hidden_states)
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = mean + z * torch.exp(0.5 * logvar)
        kld = -0.5 * torch.sum(logvar - mean.pow(2) - logvar.exp() + 1, 1).mean()

        input_emb = self.dropout(input_emb)
        token_logits, aux_logits = self.decoder(decoder_input=input_emb, latent_variable=z)

        length = corpus['target_length'] - 1

        rec_loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        rec_loss = rec_loss.reshape_as(target_text)
        rec_loss = rec_loss.sum(dim=1) / length.float()

        aux_loss = self.loss(aux_logits.view(-1, aux_logits.size(-1)), target_text.contiguous().view(-1))
        aux_loss = aux_loss.reshape_as(target_text)
        aux_loss = aux_loss.sum(dim=1) / length.float()

        kld_coef = float(epoch_idx / self.max_epoch) + 1e-3
        # gradually increase the kld weight
        loss = rec_loss.mean() + self.alpha_aux * aux_loss.mean() + kld_coef * kld
        return loss

    def calculate_nll_test(self, corpus, epoch_idx=0):
        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]
        batch_size = input_text.size(0)

        input_emb = self.token_embedder(input_text)
        hidden_states = self.encoder(input_emb)

        mean = self.hidden_to_mean(hidden_states)
        logvar = self.hidden_to_logvar(hidden_states)
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = mean + z * torch.exp(0.5 * logvar)

        token_logits, aux_logits = self.decoder(decoder_input=input_emb, latent_variable=z)

        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)
        loss = loss.sum(dim=1)
        return loss.mean()
