# @Time   : 2021/1/27
# @Author : Zhuohao Yu
# @Email  : zhuohaoyu1228@outlook.com

r"""
C2S
################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import AttributeGenerator
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder
from textbox.model.init import xavier_normal_initialization


class C2S(AttributeGenerator):
    r"""Context-aware Natural Language Generation with Recurrent Neural Network
    """

    def __init__(self, config, dataset):
        super(C2S, self).__init__(config, dataset)

        # Load hyperparameters
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_dec_layers = config['num_dec_layers']
        self.dropout_ratio = config['dropout_ratio']
        self.rnn_type = config['rnn_type']

        self.eval_generate_num = config['eval_generate_num']
        self.max_length = config['max_seq_length']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # Layers
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        self.attr_embedder = nn.ModuleList([nn.Embedding(self.attribute_size[i], self.embedding_size, padding_idx=self.padding_token_idx) for i in range(self.attribute_num)])


        self.decoder = BasicRNNDecoder(
            self.embedding_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
        )

        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.attr_linear = nn.Linear(self.embedding_size * self.attribute_num, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_ratio)

        # Loss
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        # Initialize parameters
        self.apply(xavier_normal_initialization)

    def calculate_loss(self, corpus, epoch_idx=-1, nll_test=False):
        input_text = corpus['target_idx'][:, :-1]
        input_attr = corpus['attribute_idx']
        target_text = corpus['target_idx'][:, 1:]

        attr_embeddings = []

        for attr_idx in range(self.attribute_num):
            kth_dim_attr = input_attr[:,attr_idx]
            kth_dim_embeddings = self.attr_embedder[attr_idx](kth_dim_attr)
            kth_dim_embeddings = self.dropout(kth_dim_embeddings)
            attr_embeddings.append(kth_dim_embeddings)

        attr_embeddings = torch.stack(attr_embeddings)
        attr_embeddings.permute(1, 0, 2)
        attr_embeddings = attr_embeddings.reshape(-1, self.attribute_num * self.embedding_size)

        h_c = torch.tanh(self.attr_linear(attr_embeddings))

        h_c = h_c.repeat(self.num_dec_layers, 1, 1)

        input_embeddings = self.dropout(self.token_embedder(input_text))
        outputs, hidden_states = self.decoder(input_embeddings, h_c)

        token_logits = self.vocab_linear(outputs)  # B * L * V
        token_logits = token_logits.reshape(-1, token_logits.size(-1))  #  (B * L) * V

        loss = self.loss(token_logits, target_text.reshape(-1)).reshape_as(target_text)  # B * L

        if (nll_test):
            loss = loss.sum(dim=1)
        else:
            corpus_length = corpus['target_length'] - 1
            loss = loss.sum(dim=1) / corpus_length.float()

        return loss.mean()

    def generate(self, eval_data):
        # Encoder
        attr_data = eval_data['attribute_idx'][:, :]

        attr_embeddings = []

        for attr_idx in range(self.attribute_num):
            kth_dim_attr = eval_data[:,attr_idx]
            kth_dim_embeddings = self.attr_embedder[attr_idx](kth_dim_attr)
            kth_dim_embeddings = self.dropout(kth_dim_embeddings)
            attr_embeddings.append(kth_dim_embeddings)

        attr_embeddings = torch.stack(attr_embeddings)
        attr_embeddings.permute(1, 0, 2)
        attr_embeddings = attr_embeddings.view(-1, self.attribute_num * self.embedding_size)

        h_c = torch.tanh(self.attr_linear(attr_embeddings))

        # h_c = h_c.repeat(self.num_dec_layers, 1, 1)

        generated_corpus = []
        idx2token = eval_data.idx2token

        # Decoder
        for i in range(self.eval_generate_num):
            hidden_states = h_c[i]
            hidden_states = hidden_states.repeat(self.num_dec_layers, 1)
            generated_tokens = []
            input_last = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            for _ in range(self.max_length):
                decoder_input = self.token_embedder(input_last)
                outputs, hidden_states = self.decoder(decoder_input, hidden_states)
                token_logits = self.vocab_linear(outputs)
                token_probs = F.softmax(token_logits, dim=-1).squeeze()
                token_idx = torch.multinomial(token_probs, 1)[0].item()

                if token_idx == self.eos_token_idx:
                    break
                else:
                    generated_tokens.append(idx2token[token_idx])
                    input_last = torch.LongTensor([[token_idx]]).to(self.device)

            generated_corpus.append(generated_tokens)
        return generated_corpus

    def calculate_nll_test(self, corpus, epoch_idx):
        return self.calculate_loss(corpus, epoch_idx, nll_test=True)
