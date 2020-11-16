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
from textbox.module.Encoder.vae_encoder import LSTMVAEEncoder
from textbox.module.Decoder.vae_decoder import CVAEDecoder
from textbox.model.init import xavier_normal_initialization


'''
Reference: Improved Variational Autoencoders for Text Modeling using Dilated Convolutions. ICML 2017
Code Reference: https://github.com/kefirski/contiguous-succotash
'''


class CVAE(UnconditionalGenerator):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(CVAE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.prior_size = config['hidden_size']
        # dimension for z
        # self.num_layers = config['num_layers']
        self.n_layers_decoder = config['n_layers_decoder']
        self.rnn_type = config['rnn_type']
        self.max_epoch = config['epochs']
        self.decoder_kernel_size = config['decoder_kernel_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        self.encoder = LSTMVAEEncoder(input_size=self.embedding_size,
                                      hidden_size=self.hidden_size,
                                      n_layers_encoder=config['n_layers_encoder'],
                                      n_layers_highway=config['n_layers_highway'])
        # Bidirectional LSTM encoder for LSTM VAE

        self.decoder = CVAEDecoder(decoder_kernel_size=self.decoder_kernel_size,
                                   input_size=self.embedding_size,
                                   hidden_size=self.hidden_size,
                                   drop_rate=config['decoder_drop_rate'])
        self.vocab_linear = nn.Linear(self.decoder_kernel_size[-1], self.vocab_size)
        self.hidden_to_mu = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.hidden_to_logvar = nn.Linear(2 * self.hidden_size, self.hidden_size)
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
            generate_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            for gen_idx in range(100):
                decoder_input = self.token_embedder(input_seq)
                outputs = self.decoder(decoder_input=decoder_input, z=z)
                token_logits = self.vocab_linear(outputs)
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

        outputs = self.decoder(decoder_input=input_emb, z=z)

        token_logits = self.vocab_linear(outputs)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        target_text = target_text.contiguous().view(-1)
        # print(token_logits.size(), target_text.size())
        loss = self.loss(token_logits, target_text)
        # kld_coef = (math.tanh((step - 15000) / 1000) + 1) / 2
        kld_coef = float(epoch_idx / self.max_epoch) + 1e-3
        # gradually increase the kld weight
        loss = loss + kld_coef * kld
        return loss
