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


class HRED(Seq2SeqGenerator):
    r""" This is a description
    """

    def __init__(self, config, dataset):
        super(HRED, self).__init__(config, dataset)
        # # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.bidirectional = config['bidirectional']
        self.num_directions = 2 if self.bidirectional else 1
        self.dropout_ratio = config['dropout_ratio']
        self.strategy = config['decoding_strategy']
        self.attention_type = config['attention_type']
        self.alignment_method = config['alignment_method']

        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.context_size = self.hidden_size * self.num_directions
        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)

        self.utterance_encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type, self.dropout_ratio,
            self.bidirectional
        )

        self.context_encoder = BasicRNNEncoder(
            self.hidden_size * 2, self.hidden_size, self.num_enc_layers, self.rnn_type,
            self.dropout_ratio, self.bidirectional
        )

        if self.attention_type is not None:
            self.decoder = AttentionalRNNDecoder(
                self.embedding_size + self.hidden_size, self.hidden_size, self.context_size, self.num_dec_layers, self.rnn_type,
                self.dropout_ratio, self.attention_type, self.alignment_method
            )
        else:
            self.decoder = BasicRNNDecoder(
                self.embedding_size + self.hidden_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
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
            utt_states, context_states = self.encode(batch_data) # [b, t, nd * h], [nl, b, h]
            source_length = batch_data['source_idx_length_data'] # [b, t]
            utt_masks = torch.ne(source_length, 0) # [b, t]
            
            for bid in range(utt_states.size(0)):
                encoder_states = utt_states[bid].unsqueeze(0) # [1, t, nd * h]
                decoder_states = context_states[:, bid, :].unsqueeze(1) # [nl, 1, h]
                context_state = decoder_states[-1].unsqueeze(0) # [1, 1, h]
                encoder_masks = utt_masks[bid].unsqueeze(0) # [1, t]
                
                genetare_tokens = []
                input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

                if (self.strategy == 'beam_search'):
                    hypothesis = Beam_Search_Hypothesis(
                        self.beam_size, self.sos_token_idx, self.eos_token_idx, self.device, idx2token
                    )

                for gen_idx in range(self.max_target_length):
                    input_embedding = self.token_embedder(input_seq) # [beam, 1, e]
                    decoder_input = torch.cat((input_embedding, context_state.repeat(input_embedding.size(0), 1, 1)), dim=-1) # [beam, 1, e + h]

                    if self.attention_type is not None:
                        decoder_outputs, decoder_states, _ = self.decoder(
                            decoder_input, decoder_states, encoder_states, encoder_masks
                        )
                    else:
                        decoder_outputs, decoder_states = self.decoder(decoder_input, decoder_states)

                    token_logits = self.vocab_linear(decoder_outputs)
                    if (self.strategy == 'topk_sampling'):
                        token_idx = topk_sampling(token_logits).item()
                    elif (self.strategy == 'greedy_search'):
                        token_idx = greedy_search(token_logits).item()
                    elif (self.strategy == 'beam_search'):
                        input_seq, decoder_states, encoder_states, encoder_masks = hypothesis.step(gen_idx, token_logits, decoder_states, encoder_states, encoder_masks)

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

    def encode(self, corpus):
        source_text = corpus['source_text_idx_data'] # [b, t, l]
        source_length = corpus['source_idx_length_data'] # [b, t]
        source_sentence_num = corpus['source_idx_num_data'] # [b]
        batch_size = source_text.size(0)
        turn_num = source_text.size(1)

        total_utt_states = []
        for turn in range(turn_num):
            utt_embeddings = self.token_embedder(source_text[:, turn, :]) # [b, len, e]
            utt_length = source_length[:, turn] # [b]
            utt_mask = torch.ne(utt_length, 0)
            _, utt_states = self.utterance_encoder(utt_embeddings[utt_mask], utt_length[utt_mask]) # [nl * nd, b, h]
            tp_utt_states = torch.zeros(self.num_directions * self.num_enc_layers, batch_size, self.hidden_size).to(self.device)
            tp_utt_states[:, utt_mask] = utt_states
            total_utt_states.append(tp_utt_states)

        utt_states = torch.stack(total_utt_states, dim=2) # [nl * nd, b, t, h]
        utt_states = utt_states[-self.num_directions:] # [nd, b, t, h]
        utt_states = utt_states.permute(1, 2, 0, 3).reshape(batch_size, turn_num, -1) # [b, t, nd * h]

        _, context_states = self.context_encoder(utt_states, source_sentence_num) # [nl * nd, b, h]

        if self.bidirectional:
            context_states = context_states.reshape(self.num_enc_layers, 2, batch_size, -1).sum(dim=1).contiguous() # [nl, b, h]
        return utt_states, context_states

    def calculate_loss(self, corpus, epoch_idx=0):
        utt_states, context_states = self.encode(corpus) # [b, t, nd * h], [nl, b, h]

        input_text = corpus['target_text_idx_data'][:, :-1]
        target_text = corpus['target_text_idx_data'][:, 1:]
        target_length = corpus['target_idx_length_data']
        input_embeddings = self.token_embedder(input_text) # [b, l, e]
        context_state = context_states[-1].unsqueeze(1).repeat(1, input_embeddings.size(1), 1) # [b, l, h]
        inputs = torch.cat((input_embeddings, context_state), dim=-1) # [b, l, e + h]
        
        if self.attention_type is not None:
            source_length = corpus['source_idx_length_data']
            utt_masks = torch.ne(source_length, 0)
            decoder_outputs, decoder_states, _ = self.decoder(
                inputs, context_states, utt_states, utt_masks
            )
        else:
            decoder_outputs, decoder_states = self.decoder(inputs, context_states)

        token_logits = self.vocab_linear(self.dropout(decoder_outputs)) # [b, l, v]
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        length = target_length - 1
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss
