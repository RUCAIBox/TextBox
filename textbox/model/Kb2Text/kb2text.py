# @Time   : 2021/8/23
# @Author : ZiKang Liu
# @Email  ：jason8121@foxmail.com

r"""
Graphwriter
################################################
Reference:
    Rik et al. "Text Generation from Knowledge Graphs with Graph Transformers" in ACL 2019.
"""
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from textbox.model.Knowledge.decoder import Decoder
from textbox.model.abstract_generator import AttributeGenerator
from textbox.module.Attention.attention_mechanism import MultiHeadAttention
from textbox.module.Decoder.rnn_decoder import AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Encoder.transformer_encoder import TransformerEncoder
from textbox.module.strategy import topk_sampling, greedy_search, Beam_Search_Hypothesis


class Kb2text(AttributeGenerator):
    r"""Text Generation from Knowledge Graphs with Graph Transformers.
    """

    def __init__(self, config, dataset):
        super(Kb2text, self).__init__(config, dataset)

        # load parameters info
        self.rnn_type = config['rnn_type']
        self.attention_type = config['attention_type']
        self.alignment_method = config['alignment_method']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.attn_weight_dropout_ratio = config['attn_weight_dropout_ratio']
        self.dropout_ratio = config['dropout_ratio']
        self.strategy = config['decoding_strategy']
        self.num_heads = config['num_heads']

        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        self.source_token_embedder = nn.Embedding(
            self.source_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.title_token_embedder = nn.Embedding(
            self.title_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.entity_token_embedder = nn.Embedding(
            self.entity_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.target_token_embedder = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.entity_encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, 'rnn', self.dropout_ratio
        )

        self.title_encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, 'rnn', self.dropout_ratio
        )

        self.graph_encoder = TransformerEncoder(
            TransformerEncoderLayer(self.embedding_size, self.num_heads, self.ffn_size, self.attn_dropout_ratio),
            self.num_enc_layers
        )

        self.attn_layer = MultiHeadAttention(self.embedding_size, self.num_heads, self.attn_weight_dropout_ratio)

        self.decoder = Decoder(
            self.embedding_size, self.hidden_size, self.embedding_size, self.num_dec_layers, self.rnn_type,
            self.dropout_ratio, self.attention_type, self.alignment_method
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.copy_linear = nn.Linear(self.hidden_size, 1)
        self.copy_attn = MultiHeadAttention(self.embedding_size, self.num_heads, self.attn_weight_dropout_ratio)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        self.max_target_length = config['max_seq_length']

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def encoder(self, corpus):
        # entity_text (Torch.Tensor): shape: [batch_size, length].
        entity_text = corpus['entity_idx']
        # single_entity_length (Torch.Tensor): shape: [length].
        single_entity_length = corpus['single_entity_length']
        # entity_length (Torch.Tensor): shape: [batch_size].
        entity_length = corpus['entity_length']
        # entity_type (Torch.Tensor): shape: [batch_size, length].
        entity_type = corpus['entity_type']

        # graph_mask (Torch.Tensor): shape: [batch_size].
        graph_mask = corpus['graph_mask']

        # title_idx (torch.Tensor): shape: [batch_size, length].
        title_text = corpus['title_idx']
        # title_idx (torch.Tensor): shape: [batch_size, length].
        title_length = corpus['title_idx']

        self.batch_size = entity_text.size(0)
        entity_text_mask = entity_text == self.padding_token_idx
        title_text_mask = title_text == self.padding_token_idx

        # entity encoder
        _, (entity_embeddings, _) = self.entity_encoder(
            self.entity_token_embedder(entity_text), single_entity_length
        )
        entity_embeddings = entity_embeddings[:, -2:].view(entity_embeddings.size(0), -1)
        entity_embeddings = entity_embeddings.spilt(entity_length)
        max_len = max(entity_length)
        # TODO:Concat 2 types of vertex to entity embeddings
        entity_embeddings = torch.stack([
            torch.cat([entity_embeddings[i], torch.zeros([max_len - entity_length[i], self.embedding_size])], 0) \
            for i in range(self.batch_size)], 0
        )

        # title encoder
        title_embeddings, _ = self.title_encoder(
            self.title_token_embedder(title_text), title_length
        )

        padding_mask = torch.arange(0, max_len).unsqueeze(0).repeat(max_len, 1).long()
        padding_mask = (padding_mask < entity_length.unsqueeze(1)).cuda()
        context_entity_embeddings = self.graph_encoder(entity_embeddings, graph_mask, padding_mask)

        return context_entity_embeddings, title_embeddings


def generate(self, batch_data, eval_data):
    generate_corpus = []
    idx2token = eval_data.idx2token
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
    # target_idx (torch.Tensor): shape: [batch_size].
    target_text = corpus['target_idx']
    # target_length (Torch.Tensor): shape: [batch_size]
    target_length = corpus['target_length']

    input_text = target_text[:, :-1]
    target_text = target_text[:, 1:]

    entity_embeddings, title_embeddings = self.encoder(corpus)

    input_embeddings = self.dropout(self.target_token_embedder(input_text))
    initial_vertex_embedding = torch.zeros([self.batch_size, self.hidden_size]).to(self.device)
    c = torch.zeros(self.num_dec_layers, self.batch_size, self.hidden_size).to(self.device)

    decoder_outputs, decoder_states = self.decoder(
        input_embeddings, (initial_vertex_embedding, c), entity_embeddings, title_embeddings
    )

    copy_prob = torch.sigmoid(self.copy_linear(decoder_outputs))

    vocab_pred = (1 - copy_prob) * self.vocab_linear(decoder_outputs)
    copy_pred = copy_prob * self.copy_attn(decoder_outputs, entity_embeddings)

    # token_logits (Torch.Tensor): shape: [batch_size, target_length, vocabulary_size].
    token_logits = torch.cat([vocab_pred, copy_pred], -1)

    # token_logits.view(-1, token_logits.size(-1)) (Torch.Tensor): shape: [batch_size * target_length, vocabulary_size].
    # target_text.reshape(-1) (Torch.Tensor): shape: [batch_size * target_length].
    loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.reshape(-1))
    loss = loss.reshape_as(target_text)

    loss = loss.sum(dim=1) / (target_length - 1).float()
    loss = loss.mean()
    return loss
