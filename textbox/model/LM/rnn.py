# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2021/1/2
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

r"""
RNN
################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import UnconditionalGenerator
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder
from textbox.model.init import xavier_normal_initialization


class RNN(UnconditionalGenerator):
    r""" Basic Recurrent Neural Network for Maximum Likelihood Estimation.
    """

    def __init__(self, config, dataset):
        super(RNN, self).__init__(config, dataset)

        self.sum = 0

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.dropout_ratio = config['dropout_ratio']
        self.eval_generate_num = config['eval_generate_num']
        self.max_length = config['max_seq_length']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        self.decoder = BasicRNNDecoder(
            self.embedding_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, batch_data, eval_data):
        generate_corpus = []
        idx2token = eval_data.idx2token
        hidden_states = torch.zeros(self.num_dec_layers, 1, self.hidden_size).to(self.device)
        generate_tokens = []
        input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
        for gen_idx in range(self.max_length):
            decoder_input = self.token_embedder(input_seq)
            outputs, hidden_states = self.decoder(decoder_input, hidden_states)
            token_logits = self.vocab_linear(outputs)
            token_probs = F.softmax(token_logits, dim=-1).squeeze()
            token_idx = torch.multinomial(token_probs, 1)[0].item()

            if token_idx == self.eos_token_idx:
                break
            else:
                generate_tokens.append(idx2token[token_idx])
                input_seq = torch.LongTensor([[token_idx]]).to(self.device)
        generate_corpus.append(generate_tokens)
        return generate_corpus

    def forward(self, corpus, epoch_idx=-1, nll_test=False):
        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]

        self.sum += 1
        # print ("in rnn: ", torch.distributed.get_rank(), self.sum, input_text.shape, target_text.shape)

        input_embeddings = self.dropout(self.token_embedder(input_text))
        outputs, hidden_states = self.decoder(input_embeddings)

        token_logits = self.vocab_linear(outputs)
        token_logits = token_logits.view(-1, token_logits.size(-1))

        loss = self.loss(token_logits, target_text.contiguous().view(-1)).reshape_as(target_text)
        if (nll_test):
            loss = loss.sum(dim=1)
        else:
            length = corpus['target_length'] - 1
            loss = loss.sum(dim=1) / length.float()
        return loss.mean()

    def calculate_nll_test(self, corpus, epoch_idx):
        return self.calculate_loss(corpus, epoch_idx, nll_test=True)
