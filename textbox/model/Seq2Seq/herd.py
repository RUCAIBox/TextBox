# @Time   : 2021/1/26
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

r"""
HierarchicalRNN
################################################
Reference:
    Serban et al. "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models" in AAAI 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling, greedy_search, Beam_Search_Hypothesis


class HERD(Seq2SeqGenerator):
    r""" This is a description
    """

    def __init__(self, config, dataset):
        super(HERD, self).__init__(config, dataset)
        # # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.alignment_method = config['alignment_method']
        self.context_size = config['context_size']
        self.strategy = config['decoding_strategy']
        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        self.token_encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type, self.dropout_ratio,
            self.bidirectional
        )

        self.session_encoder = BasicRNNEncoder(
            self.hidden_size * self.num_enc_layers, self.hidden_size, self.num_enc_layers, self.rnn_type,
            self.dropout_ratio, self.bidirectional
        )

        self.decoder = BasicRNNDecoder(
            self.embedding_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        self.max_target_length = config['max_seq_length']

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, eval_dataloader):
        generate_corpus = []
        idx2token = eval_dataloader.idx2token

        for batch_data in eval_dataloader:
            source_text = batch_data['source_text_idx_data']
            source_sentence_num = batch_data['source_idx_num_data']
            source_length = batch_data['source_idx_length_data']

            token_masks = torch.ne(source_length, 0)
            total_token_encoder = []
            for turn in range(source_text.size(1)):
                source_embeddings = self.token_embedder(source_text[:, turn, :])
                token_encoder_outputs, token_encoder_states = self.token_encoder(
                    source_embeddings[token_masks[:, turn]], source_length[:, turn][token_masks[:, turn]]
                )
                if self.bidirectional:
                    token_encoder_states_process = torch.full(
                        (2 * self.num_enc_layers, source_text.size(0), self.hidden_size), self.padding_token_idx
                    ).float()
                else:
                    token_encoder_states_process = torch.full(
                        (self.num_enc_layers, source_text.size(0), self.hidden_size), self.padding_token_idx
                    ).float()
                token_encoder_states_process[:, token_masks[:, turn]] = token_encoder_states

                if self.bidirectional:
                    token_encoder_states_process = token_encoder_states_process[::2]
                total_token_encoder.append(token_encoder_states_process)

            total_token_encoder = torch.cat(total_token_encoder, dim=0)
            total_token_encoder = total_token_encoder.view(source_text.size(0), source_length.size(1), -1)
            session_encoder_outputs, session_encoder_states = self.session_encoder(
                total_token_encoder, source_sentence_num
            )

            if self.bidirectional:
                session_encoder_outputs = session_encoder_outputs[:, :, self.hidden_size:
                                                                  ] + session_encoder_outputs[:, :, :self.hidden_size]
                session_encoder_states = session_encoder_states[::2]

            for bid in range(source_text.size(0)):
                decoder_states = session_encoder_states[:, bid, :].unsqueeze(1)
                session_encoder_output = session_encoder_outputs[bid, :, :].unsqueeze(0)
                genetare_tokens = []
                input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

                if (self.strategy == 'beam_search'):
                    hypothesis = Beam_Search_Hypothesis(
                        self.beam_size, self.sos_token_idx, self.eos_token_idx, self.device, idx2token
                    )

                for gen_idx in range(self.max_target_length):
                    decoder_input = self.token_embedder(input_seq)
                    decoder_outputs, decoder_states = self.decoder(decoder_input, decoder_states)
                    token_logits = self.vocab_linear(decoder_outputs)
                    if (self.strategy == 'topk_sampling'):
                        token_idx = topk_sampling(token_logits).item()
                    elif (self.strategy == 'greedy_search'):
                        token_idx = greedy_search(token_logits).item()
                    elif (self.strategy == 'beam_search'):
                        input_seq, decoder_states = hypothesis.step(gen_idx, token_logits, decoder_states)

                    if (self.strategy in ['topk_sampling', 'geedy_search']):
                        if token_idx == self.eos_token_idx:
                            break
                        else:
                            genetare_tokens.append(idx2token[token_idx])
                            input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                    elif (self.strategy == 'beam_search'):
                        if (hypothesis.stop()):
                            break

                if (self.strategy == 'beam_search'):
                    generate_tokens = hypothesis.generate()

                generate_corpus.append(generate_tokens)

        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        source_text = corpus['source_text_idx_data']
        source_sentence_num = corpus['source_idx_num_data']
        source_length = corpus['source_idx_length_data']

        input_text = corpus['target_text_idx_data'][:, :-1]
        target_text = corpus['target_text_idx_data'][:, 1:]
        target_length = corpus['target_idx_length_data']

        token_masks = torch.ne(source_length, 0)

        total_token_encoder = []
        for turn in range(source_text.size(1)):
            source_embeddings = self.dropout(self.token_embedder(source_text[:, turn, :]))
            token_encoder_outputs, token_encoder_states = self.token_encoder(
                source_embeddings[token_masks[:, turn]], source_length[:, turn][token_masks[:, turn]]
            )

            if self.bidirectional:
                token_encoder_states_process = torch.full(
                    (2 * self.num_enc_layers, source_text.size(0), self.hidden_size), self.padding_token_idx
                ).float()
            else:
                token_encoder_states_process = torch.full((self.num_enc_layers, source_text.size(0), self.hidden_size),
                                                          self.padding_token_idx).float()
            token_encoder_states_process[:, token_masks[:, turn]] = token_encoder_states

            if self.bidirectional:
                token_encoder_states_process = token_encoder_states_process[::2]
            total_token_encoder.append(token_encoder_states_process)

        total_token_encoder = torch.cat(total_token_encoder, dim=0)
        total_token_encoder = total_token_encoder.view(source_text.size(0), source_length.size(1), -1)
        session_encoder_outputs, session_encoder_states = self.session_encoder(total_token_encoder, source_sentence_num)

        if self.bidirectional:
            session_encoder_outputs = session_encoder_outputs[:, :,
                                                              self.hidden_size:] + session_encoder_outputs[:, :, :self.
                                                                                                           hidden_size]
            session_encoder_states = session_encoder_states[::2]

        input_embeddings = self.dropout(self.token_embedder(input_text))
        decoder_outputs, decoder_states = self.decoder(input_embeddings, session_encoder_states)

        token_logits = self.vocab_linear(decoder_outputs)
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        length = target_length - 1
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss
