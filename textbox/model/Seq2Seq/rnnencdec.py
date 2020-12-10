# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import ConditionalGenerator
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling


class RNNEncDec(ConditionalGenerator):
    r"""RNN-based Encoder-Decoder architecture is a basic framework for conditional text generation.

    """

    def __init__(self, config, dataset):
        super(RNNEncDec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.attention_type = config['attention_type']
        self.alignment_method = config['alignment_method']
        self.context_size = config['context_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.source_token_embedder = nn.Embedding(self.source_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        
        if config['share_vocab']:
            self.target_token_embedder = self.source_token_embedder
        else:
            self.target_token_embedder = nn.Embedding(self.target_vocab_size, self.embedding_size,
                                                      padding_idx=self.padding_token_idx)
        
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type,
                                       self.dropout_ratio, self.bidirectional)

        if self.attention_type is not None:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.hidden_size, self.context_size,
                                                 self.num_dec_layers, self.rnn_type, self.dropout_ratio,
                                                 self.attention_type, self.alignment_method)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.hidden_size, self.num_dec_layers,
                                           self.rnn_type, self.dropout_ratio)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.target_vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx)
        
        self.max_target_length = config['target_max_seq_length']

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, eval_dataloader):
        generate_corpus = []
        idx2token = eval_dataloader.target_idx2token
        for batch_data in eval_dataloader:
            source_text = batch_data['source_idx']
            source_length = batch_data['source_length']
            source_embeddings = self.source_token_embedder(source_text)
            encoder_outputs, encoder_states = self.encoder(source_embeddings, source_length)

            if self.bidirectional:
                encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
                encoder_states = encoder_states[0:encoder_states.size(0):2]
                
            encoder_masks = torch.ne(source_text, self.padding_token_idx)
            for bid in range(source_text.size(0)):
                decoder_states = encoder_states[:, bid, :].unsqueeze(1)
                generate_tokens = []
                input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
                for gen_idx in range(self.max_target_length):
                    decoder_input = self.target_token_embedder(input_seq)
                    if self.attention_type is not None:
                        decoder_outputs, decoder_states, _ = self.decoder(decoder_input,
                                                                          decoder_states,
                                                                          encoder_outputs[bid, :, :].unsqueeze(0),
                                                                          encoder_masks[bid, :].unsqueeze(0))
                    else:
                        decoder_outputs, decoder_states = self.decoder(decoder_input,
                                                                       decoder_states)
                    token_logits = self.vocab_linear(decoder_outputs)
                    token_idx = topk_sampling(token_logits)
                    token_idx = token_idx.item()
                    # print(token_idx)
                    if token_idx == self.eos_token_idx:
                        break
                    else:
                        generate_tokens.append(idx2token[token_idx])
                        input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                generate_corpus.append(generate_tokens)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        source_text = corpus['source_idx']
        source_length = corpus['source_length']

        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]
        # print(type(source_text), type(source_length))

        source_embeddings = self.dropout(self.source_token_embedder(source_text))
        input_embeddings = self.dropout(self.target_token_embedder(input_text))
        # print(source_embeddings.device, input_embeddings.device, self.encoder.device)
        encoder_outputs, encoder_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
            encoder_states = encoder_states[0:encoder_states.size(0):2]

        encoder_masks = torch.ne(source_text, self.padding_token_idx)

        if self.attention_type is not None:
            decoder_outputs, decoder_states, _ = self.decoder(input_embeddings, encoder_states, encoder_outputs,
                                                              encoder_masks)
        else:
            decoder_outputs, decoder_states = self.decoder(input_embeddings, encoder_states)

        token_logits = self.vocab_linear(decoder_outputs)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        target_text = target_text.contiguous().view(-1)

        loss = self.loss(token_logits, target_text)
        return loss
