# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import ConditionalGenerator
from textbox.module.Encoder.transformer_encoder import TransformerEncoder
from textbox.module.Decoder.transformer_decoder import TransformerDecoder
from textbox.model.init import xavier_normal_initialization


class TransformerEncDec(ConditionalGenerator):
    r"""RNN-based Encoder-Decoder architecture is a basic framework for conditional text generation.

    """
    input_type = InputType.PAIRTEXT

    def __init__(self, config, dataset):
        super(TransformerEncDec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.ffn_size = config['ffn_size']
        self.num_heads = config['num_heads']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.attn_dropout_ratio = config['attn_dropout_ratio']
        self.attn_weight_dropout_ratio = config['attn_weight_dropout_ratio']
        self.ffn_dropout_ratio = config['ffn_dropout_ratio']
        self.ffn_activation_func = config['ffn_activation_func']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.source_token_embedder = nn.Embedding(self.source_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.target_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.position_embedder = SinusoidalPositionalEmbedding(self.embedding_size)
        self.self_attn_mask = SelfAttentionMask()

        self.encoder = TransformerEncoder(self.embedding_size, self.ffn_size, self.num_enc_layers, self.num_heads,
                                          self.attn_dropout_ratio, self.attn_weight_dropout_ratio,
                                          self.ffn_dropout_ratio, self.ffn_activation_func)

        self.decoder = TransformerDecoder(self.embedding_size, self.ffn_size, self.num_dec_layers, self.num_heads,
                                          self.attn_dropout_ratio, self.attn_weight_dropout_ratio,
                                          self.ffn_dropout_ratio, self.ffn_activation_func, with_external=True)

        self.vocab_linear = nn.Linear(self.embedding_size, self.target_vocab_size)
        self.vocab_linear.weight = self.target_token_embedder.weight

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, eval_data):
        generate_corpus = []
        number_to_gen = 10
        idx2token = eval_data.idx2token
        for _ in range(number_to_gen):
            hidden_states = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
            generate_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            for gen_idx in range(100):
                decoder_input = self.token_embedder(input_seq)
                outputs, hidden_states = self.decoder(hidden_states, decoder_input)
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
        return generate_corpus

    def calculate_loss(self, corpus):
        source_text = corpus['source_idx']

        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]

        source_embeddings = self.source_token_embedder(source_text) + self.position_embedder(source_text).to(self.device)
        source_padding_mask = torch.eq(source_text, self.padding_token_idx).to(self.device)
        encoder_outputs = self.encoder(source_embeddings, self_padding_mask=source_padding_mask,
                                       output_all_encoded_layers=False)

        input_embeddings = self.target_token_embedder(input_text) + self.position_embedder(input_text).to(self.device)
        self_padding_mask = torch.eq(input_text, self.padding_token_idx).to(self.device)
        self_attn_mask = self.self_attn_mask(input_text.size(-1)).bool().to(self.device)
        decoder_outputs = self.decoder(input_embeddings,
                                       self_padding_mask=self_padding_mask,
                                       self_attn_mask=self_attn_mask,
                                       external_states=encoder_outputs,
                                       external_padding_mask=source_padding_mask)

        token_logits = self.vocab_linear(decoder_outputs)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        target_text = target_text.contiguous().view(-1)

        loss = self.loss(token_logits, target_text)
        return loss
