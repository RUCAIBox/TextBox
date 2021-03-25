# @Time   : 2021/1/27
# @Author : Zhuohao Yu
# @Email  : zhuohao@ruc.edu.cn

r"""
C2S
################################################
Reference:
    Jian Tang et al. "Context-aware Natural Language Generation with Recurrent Neural Networks" in 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import AttributeGenerator
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import Beam_Search_Hypothesis


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
        self.is_gated = config['gated']
        self.decoding_strategy = config['decoding_strategy']

        if self.decoding_strategy == 'beam_search':
            self.beam_size = config['beam_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # Layers
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        self.attr_embedder = nn.ModuleList([
            nn.Embedding(self.attribute_size[i], self.embedding_size) for i in range(self.attribute_num)
        ])

        self.decoder = BasicRNNDecoder(
            self.embedding_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
        )

        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.attr_linear = nn.Linear(self.attribute_num * self.embedding_size, self.hidden_size)

        if self.is_gated:
            self.gate_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_ratio)

        # Loss
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        # Initialize parameters
        self.apply(xavier_normal_initialization)

    def encoder(self, attr_data):
        attr_embeddings = []

        for attr_idx in range(self.attribute_num):
            kth_dim_attr = attr_data[:, attr_idx]
            kth_dim_embeddings = self.attr_embedder[attr_idx](kth_dim_attr)
            attr_embeddings.append(kth_dim_embeddings)

        attr_embeddings = torch.cat(attr_embeddings, dim=1)
        h_c = torch.tanh(self.attr_linear(attr_embeddings)).contiguous()
        return attr_embeddings, h_c

    def forward(self, corpus, epoch_idx=-1, nll_test=False):
        input_text = corpus['target_idx'][:, :-1]
        input_attr = corpus['attribute_idx']
        target_text = corpus['target_idx'][:, 1:]

        attr_embeddings, h_c_1D = self.encoder(input_attr)

        h_c = h_c_1D.repeat(self.num_dec_layers, 1, 1)
        input_embeddings = self.token_embedder(input_text)
        outputs, _ = self.decoder(input_embeddings, h_c)

        if self.is_gated:
            m_t = torch.sigmoid(self.gate_linear(outputs)).permute(1, 0, 2)
            outputs = outputs + (m_t * h_c_1D).permute(1, 0, 2)

        outputs = self.dropout(outputs)

        token_logits = self.vocab_linear(outputs)
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        length = corpus['target_length'] - 1
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss

    def generate(self, batch_data, eval_data):
        generate_corpus = []
        idx2token = eval_data.idx2token
        batch_size = batch_data['attribute_idx'].size(0)
        attr_embeddings, h_c_1D = self.encoder(batch_data['attribute_idx'])
        h_c = h_c_1D.repeat(self.num_dec_layers, 1, 1)

        for bid in range(batch_size):
            hidden_states = h_c[:, bid, :].unsqueeze(1).contiguous()
            generate_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

            if (self.decoding_strategy == 'beam_search'):
                hypothesis = Beam_Search_Hypothesis(
                    self.beam_size, self.sos_token_idx, self.eos_token_idx, self.device, idx2token
                )

            for gen_idx in range(self.max_length):
                decoder_input = self.token_embedder(input_seq)
                outputs, hidden_states = self.decoder(decoder_input, hidden_states)
                if self.is_gated:
                    m_t = torch.sigmoid(self.gate_linear(outputs))
                    outputs = outputs + m_t * h_c_1D[bid]
                token_logits = self.vocab_linear(outputs)

                if (self.decoding_strategy == 'topk_sampling'):
                    token_idx = topk_sampling(token_logits).item()
                elif (self.decoding_strategy == 'greedy_search'):
                    token_idx = greedy_search(token_logits).item()
                elif (self.decoding_strategy == 'beam_search'):
                    input_seq, hidden_states = \
                        hypothesis.step(gen_idx, token_logits, hidden_states)

                if (self.decoding_strategy in ['topk_sampling', 'greedy_search']):
                    if token_idx == self.eos_token_idx:
                        break
                    else:
                        generate_tokens.append(idx2token[token_idx])
                        input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                elif (self.decoding_strategy == 'beam_search'):
                    if (hypothesis.stop()):
                        break

            if (self.decoding_strategy == 'beam_search'):
                generate_tokens = hypothesis.generate()

            generate_corpus.append(generate_tokens)

        return generate_corpus
