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
        self.is_gated = config['gated']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # Layers
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        self.attr_embedder = nn.ModuleList([
            nn.Embedding(self.attribute_size[i], min(self.embedding_size, self.attribute_size[i]), padding_idx=self.padding_token_idx)
            for i in range(self.attribute_num)
        ])

        total_emb_size = 0
        for i in range(self.attribute_num):
            total_emb_size += min(self.embedding_size, self.attribute_size[i])

        self.decoder = BasicRNNDecoder(
            self.embedding_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
        )

        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.attr_linear = nn.Linear(total_emb_size, self.hidden_size * self.num_dec_layers)
        
        if self.is_gated:
            self.gate_hc_linear = nn.Linear(total_emb_size, self.hidden_size)
            self.gate_linear = nn.Linear(self.hidden_size, self.hidden_size)

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
            kth_dim_attr = input_attr[:, attr_idx]
            kth_dim_embeddings = self.attr_embedder[attr_idx](kth_dim_attr)
            attr_embeddings.append(kth_dim_embeddings)

        attr_embeddings = torch.cat(attr_embeddings, dim=1)

        h_c = torch.relu(self.attr_linear(attr_embeddings))
        
        h_c = h_c.reshape(-1, self.num_dec_layers, self.hidden_size)
        h_c = h_c.permute(1, 0, 2).contiguous()

        input_embeddings = self.token_embedder(input_text)
        outputs, hidden_states = self.decoder(input_embeddings, h_c)
        
        # print("outputs", outputs.shape)
        if self.is_gated:
            h_c_1D = torch.relu(self.gate_hc_linear(attr_embeddings))
            m_t = torch.relu(self.gate_linear(outputs)).permute(1, 0, 2)
            m_t = (m_t * h_c_1D).permute(1, 0, 2)
            outputs = torch.add(outputs, m_t)

        outputs = self.dropout(outputs)

        token_logits = self.vocab_linear(outputs)
        token_logits = token_logits.view(-1, token_logits.size(-1))

        loss = self.loss(token_logits, target_text.contiguous().view(-1)).reshape_as(target_text)
        if (nll_test):
            loss = loss.sum(dim=1)
        else:
            length = corpus['target_length'] - 1
            loss = loss.sum(dim=1) / length.float()
        return loss.mean()

    def generate_for_corpus(self, eval_data, corpus):

        attr_data = corpus['attribute_idx']

        attr_embeddings = []

        for attr_idx in range(self.attribute_num):
            kth_dim_attr = attr_data[:, attr_idx]
            kth_dim_embeddings = self.attr_embedder[attr_idx](kth_dim_attr)
            attr_embeddings.append(kth_dim_embeddings)

        attr_embeddings = torch.cat(attr_embeddings, dim=1)

        h_c = torch.relu(self.attr_linear(attr_embeddings)).contiguous()

        if self.is_gated:
            h_c_1D = torch.relu(self.gate_hc_linear(attr_embeddings))

        generated_corpus = []
        idx2token = eval_data.idx2token

        # Decoder
        # hidden_states = h_c
        # hidden_states = hidden_states.repeat(self.num_dec_layers, 1, 1)

        cur_batch_size = len(h_c)

        for data_idx in range(cur_batch_size):
            generated_tokens = []
            input_last = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            hidden_states = h_c[data_idx].to(self.device)
            # hidden_states = hidden_states.repeat(self.num_dec_layers, 1).to(self.device)
            hidden_states = hidden_states.reshape(self.num_dec_layers, 1, self.hidden_size).contiguous()
            for _ in range(self.max_length):
                decoder_input = self.token_embedder(input_last)
                outputs, hidden_states = self.decoder(decoder_input, hidden_states)
                
                if self.is_gated:
                    m_t = torch.relu(self.gate_linear(outputs)) * h_c_1D[data_idx]
                    outputs = torch.add(outputs, m_t)

                token_logits = self.vocab_linear(outputs)
                token_probs = F.softmax(token_logits, dim=-1).squeeze()
                # token_idx = torch.multinomial(token_probs, 1)[0].item()
                token_idx = torch.argmax(token_probs).item()

                if token_idx == self.eos_token_idx:
                    break
                else:
                    generated_tokens.append(idx2token[token_idx])
                    input_last = torch.LongTensor([[token_idx]]).to(self.device)
            generated_corpus.append(generated_tokens)

        print(generated_corpus)

        return generated_corpus

    def generate(self, eval_data):
        # Encoder
        generated_corpus = []
        indx = 0
        for corpus in eval_data:
            indx = indx + 1
            print("Corpus #", indx)
            generated_corpus += self.generate_for_corpus(eval_data, corpus)
            print("-" * 100)

        return generated_corpus

    def calculate_nll_test(self, corpus, epoch_idx):
        return self.calculate_loss(corpus, epoch_idx, nll_test=True)
