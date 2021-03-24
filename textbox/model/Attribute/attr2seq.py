# @Time   : 2021/2/3
# @Author : Zhipeng Chen
# @Email  : zhipeng_chen@ruc.edu.cn

r"""
Attr2Seq
################################################
Reference:
    Li Dong et al. "Learning to Generate Product Reviews from Attributes" in 2017.
"""

import torch
import torch.nn as nn

from textbox.model.abstract_generator import AttributeGenerator
from textbox.module.Decoder.rnn_decoder import AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling, greedy_search, Beam_Search_Hypothesis


class Attr2Seq(AttributeGenerator):
    r"""Attribute Encoder and RNN-based Decoder architecture is a basic frame work for Attr2Seq text generation.
    """

    def __init__(self, config, dataset):
        super(Attr2Seq, self).__init__(config, dataset)

        # load parameters info
        self.rnn_type = config['rnn_type']
        self.attention_type = config['attention_type']
        self.alignment_method = config['alignment_method']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_dec_layers = config['num_dec_layers']
        self.dropout_ratio = config['dropout_ratio']
        self.strategy = config['decoding_strategy']

        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        self.source_token_embedder = nn.ModuleList([
            nn.Embedding(self.attribute_size[i], self.embedding_size) for i in range(self.attribute_num)
        ])
        self.target_token_embedder = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.decoder = AttentionalRNNDecoder(
            self.embedding_size, self.hidden_size, self.embedding_size, self.num_dec_layers, self.rnn_type,
            self.dropout_ratio, self.attention_type, self.alignment_method
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        self.max_target_length = config['max_seq_length']

        self.H = nn.Linear(self.attribute_num * self.embedding_size, self.num_dec_layers * self.hidden_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def encoder(self, source_idx):
        r""" 
            Args:
                source_idx (Torch.Tensor): source attribute index, shape: [batch_size, attribute_num].
            
            Returns:
                tuple:
                    - Torch.Tensor: output features, shape: [batch_size, attribute_num, embedding_size].
                    - Torch.Tensor: hidden states, shape: [num_dec_layers, batch_size, hidden_size].
        """
        # g (torch.Tensor): [batch_size, attribute_num * embedding_size].
        g = [self.source_token_embedder[i](source_idx[:, i]) for i in range(self.attribute_num)]
        g = torch.cat(g, 1)

        #outputs (Torch.Tensor): shape: [batch_size, attribute_num, embedding_size].
        outputs = g.reshape(self.batch_size, self.attribute_num, self.embedding_size)

        # a (Torch.Tensor): shape: [batch_size, num_dec_layers * hidden_size].
        a = torch.tanh(self.H(g))

        # hidden_states (Torch.Tensor): shape: [num_dec_layers, batch_size, hidden_size].
        hidden_states = a.reshape(self.batch_size, self.num_dec_layers, self.hidden_size)
        hidden_states = hidden_states.transpose(0, 1)

        return outputs, hidden_states

    def generate(self, eval_dataloader):
        generate_corpus = []
        idx2token = eval_dataloader.idx2token

        for batch_data in eval_dataloader:
            source_idx = batch_data['attribute_idx']
            self.batch_size = source_idx.size(0)

            encoder_outputs, encoder_states = self.encoder(source_idx)

            for bid in range(self.batch_size):
                c = torch.zeros(self.num_dec_layers, 1, self.hidden_size).to(self.device)
                decoder_states = (encoder_states[:, bid, :].unsqueeze(1), c)
                encoder_output = encoder_outputs[bid, :, :].unsqueeze(0)
                generate_tokens = []
                input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

                if (self.strategy == 'beam_search'):
                    hypothesis = Beam_Search_Hypothesis(
                        self.beam_size, self.sos_token_idx, self.eos_token_idx, self.device, idx2token
                    )

                for gen_idx in range(self.max_target_length):
                    decoder_input = self.target_token_embedder(input_seq)
                    decoder_outputs, decoder_states, _ = self.decoder(decoder_input, decoder_states, encoder_output)

                    token_logits = self.vocab_linear(decoder_outputs)
                    if (self.strategy == 'topk_sampling'):
                        token_idx = topk_sampling(token_logits).item()
                    elif (self.strategy == 'greedy_search'):
                        token_idx = greedy_search(token_logits).item()
                    elif (self.strategy == 'beam_search'):
                        input_seq, decoder_states, encoder_output = \
                            hypothesis.step(gen_idx, token_logits, decoder_states, encoder_output)

                    if (self.strategy in ['topk_sampling', 'greedy_search']):
                        if token_idx == self.eos_token_idx:
                            break
                        else:
                            generate_tokens.append(idx2token[token_idx])
                            input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                    elif (self.strategy == 'beam_search'):
                        if (hypothesis.stop()):
                            break

                if (self.strategy == 'beam_search'):
                    generate_tokens = hypothesis.generate()

                generate_corpus.append(generate_tokens)

        return generate_corpus

    def forward(self, corpus, epoch_idx=0):
        # target_length (Torch.Tensor): shape: [batch_size]
        target_length = corpus['target_length']
        # attribute_idx (Torch.Tensor): shape: [batch_size, attribute_num].
        attribute_idx = corpus['attribute_idx']
        # target_idx (torch.Tensor): shape: [batch_size, length].
        target_idx = corpus['target_idx']
        self.batch_size = attribute_idx.size(0)

        encoder_outputs, encoder_states = self.encoder(attribute_idx)

        input_text = target_idx[:, :-1]
        target_text = target_idx[:, 1:]
        input_embeddings = self.dropout(self.target_token_embedder(input_text))

        c = torch.zeros(self.num_dec_layers, self.batch_size, self.hidden_size).to(self.device)
        decoder_outputs, decoder_states, _ = \
            self.decoder(input_embeddings, (encoder_states.contiguous(), c), encoder_outputs)

        # token_logits (Torch.Tensor): shape: [batch_size, target_length, vocabulary_size].
        token_logits = self.vocab_linear(decoder_outputs)

        # token_logits.view(-1, token_logits.size(-1)) (Torch.Tensor): shape: [batch_size * target_length, vocabulary_size].
        # target_text.reshape(-1) (Torch.Tensor): shape: [batch_size * target_length].
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.reshape(-1))
        loss = loss.reshape_as(target_text)

        loss = loss.sum(dim=1) / (target_length - 1).float()
        loss = loss.mean()
        return loss
