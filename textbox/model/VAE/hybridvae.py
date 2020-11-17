# @Time   : 2020/11/9
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from textbox.utils import InputType
from textbox.model.abstract_generator import UnconditionalGenerator
# from recbole.model.loss import BPRLoss
from textbox.module.Encoder.vae_encoder import HybridEncoder
from textbox.module.Decoder.vae_decoder import HybridDecoder
from textbox.model.init import xavier_normal_initialization


'''
Reference: A Hybrid Convolutional Variational Autoencoder for Text Generation. EMNLP 2017
Code Reference: https://github.com/kefirski/hybrid_rvae
'''


class HybridVAE(UnconditionalGenerator):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(HybridVAE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = 512 # config['hidden_size']
        self.prior_size = config['hidden_size']
        # dimension for z
        self.num_layers = config['num_layers']
        self.max_epoch = config['epochs']
        self.alpha_aux = config['alpha_aux']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        self.encoder = HybridEncoder(input_size=self.embedding_size,
                                     hidden_size=self.hidden_size)
        # Bidirectional LSTM encoder for LSTM VAE

        self.decoder = HybridDecoder(vocab_size=self.vocab_size, hidden_size=self.hidden_size,
                                     num_layers=self.num_layers, embedding_size=self.embedding_size)
        # self.vocab_linear = nn.Linear(self.decoder_kernel_size[-1], self.vocab_size)
        self.hidden_to_mu = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.step = 0

    def generate(self, eval_data):
        generate_corpus = []
        number_to_gen = 10
        idx2token = eval_data.idx2token
        for _ in range(number_to_gen):
            z = torch.randn(size=(1, self.hidden_size), device=self.device)
            cnn_out = self.decoder.conv_decoder(z)
            hidden_states = torch.randn(size=(self.num_layers, 1, self.hidden_size), device=self.device)
            generate_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            cnn_out = torch.transpose(cnn_out, 0, 1).contiguous()
            # print(cnn_out.size())
            for gen_idx in range(100):
                decoder_input = self.token_embedder(input_seq)
                # token_logits, aux_logits = self.decoder(decoder_input=decoder_input, latent_variable=z)
                token_logits, hidden_states = self.decoder.rnn_decoder(cnn_out[gen_idx].unsqueeze(1),
                                                                       decoder_input=decoder_input,
                                                                       initial_state=hidden_states)
                # token_logits = self.vocab_linear(outputs)
                topv, topi = torch.log(F.softmax(token_logits, dim=-1) + 1e-12).data.topk(k=4)
                topi = topi.squeeze()
                token_idx = topi[0].item()
                if token_idx == self.eos_token_idx or gen_idx >= 100:
                    break
                else:
                    generate_tokens.append(idx2token[token_idx])
                    input_seq = torch.LongTensor([[token_idx]]).to(self.device)
            generate_corpus.append(generate_tokens)
        for gen_seq in generate_corpus[:10]:
            print(gen_seq)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]
        batch_size = input_text.size(0)

        input_emb = self.token_embedder(input_text)

        encoder_hidden = self.encoder(input_emb)
        # print(encoder_hidden.size())
        mu = self.hidden_to_mu(encoder_hidden)
        logvar = self.hidden_to_logvar(encoder_hidden)
        z = torch.randn([batch_size, self.hidden_size]).to(self.device)
        z = mu + z * torch.exp(0.5 * logvar)
        kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()

        token_logits, aux_logits = self.decoder(decoder_input=input_emb, latent_variable=z)
        # print(token_logits.size(), aux_logits.size())

        # token_logits = self.vocab_linear(outputs)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        aux_logits = aux_logits.view(-1, aux_logits.size(-1))
        target_text = target_text.contiguous().view(-1)
        # print(token_logits.size(), target_text.size())
        rec_loss = self.loss(token_logits, target_text)
        aux_loss = self.loss(aux_logits, target_text)
        # kld_coef = (math.tanh((step - 15000) / 1000) + 1) / 2
        kld_coef = float(epoch_idx / self.max_epoch) + 1e-3
        # gradually increase the kld weight
        loss = rec_loss + self.alpha_aux * aux_loss + kld_coef * kld
        return loss
