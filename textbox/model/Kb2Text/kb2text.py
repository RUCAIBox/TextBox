# @Time   : 2021/8/23
# @Author : ZiKang Liu
# @Email  ：jason8121@foxmail.com

r"""
Graphwriter
################################################
Reference:
    Rik et al. "Text Generation from Knowledge Graphs with Graph Transformers" in ACL 2019.
"""
import dgl
import torch
from torch import nn
from torch.nn.modules import module
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from textbox.data.utils import pad_sequence
from textbox.model.Kb2Text.GAT import GraphTransformer
from textbox.model.Kb2Text.decoder import Decoder
from textbox.model.abstract_generator import AttributeGenerator, Seq2SeqGenerator
from textbox.module.Attention.attention_mechanism import MultiHeadAttention
from textbox.model.init import xavier_normal_initialization
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Encoder.transformer_encoder import TransformerEncoder
from textbox.module.strategy import topk_sampling, greedy_search, Beam_Search_Hypothesis


class Kb2Text(Seq2SeqGenerator):
    r"""Text Generation from Knowledge Graphs with Graph Transformers.
    """

    def __init__(self, config, dataset):
        super(Kb2Text, self).__init__(config, dataset)

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
        self.GAT_layer_nums = config['GAT_layer_nums']
        self.NODE_TYPE = {'entity': 0, 'root': 1, 'relation': 2}
        self.REL_SET = ['USED-FOR', 'CONJUNCTION', 'FEATURE-OF', 'PART-OF', 'COMPARE', 'EVALUATE-FOR',
                        'HYPONYM-OF']
        self.source_relation_idx2token = dataset.source_relation_idx2token
        self.source_relation_token2idx = dataset.source_relation_token2idx
        self.ALL_REL_SET = ['root', 'USED-FOR', 'CONJUNCTION', 'FEATURE-OF', 'PART-OF', 'COMPARE', 'EVALUATE-FOR',
                            'HYPONYM-OF', 'USER-FOR-INV', 'CONJUNCTION-INV', 'FEATURE-OF-INV', 'PART-OF-INV',
                            'COMPARE-INV', 'EVALUATE-FOR-INV', 'HYPNOYM-OF-INV']

        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx
        self.source_token2idx = dataset.source_token2idx
        self.source_entity_token2idx = dataset.source_entity_token2idx

        self.source_token_embedder = nn.Embedding(
            self.source_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.target_token_embedder = nn.Embedding(
            self.target_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.rel_token_embedder = nn.Embedding(
            len(self.ALL_REL_SET), self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.entity_encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size // 2, self.num_enc_layers, 'lstm', self.dropout_ratio
        )

        self.source_encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size // 2, self.num_enc_layers, 'lstm', self.dropout_ratio
        )

        self.graph_encoder = GraphTransformer(self.embedding_size, self.attn_weight_dropout_ratio, self.dropout_ratio,
                                              self.GAT_layer_nums)

        self.attn_layer = MultiHeadAttention(self.embedding_size, self.num_heads, self.attn_weight_dropout_ratio)

        self.decoder = Decoder(
            self.embedding_size, self.hidden_size, self.embedding_size, self.num_dec_layers, self.rnn_type,
            self.dropout_ratio, self.attention_type, self.alignment_method
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(3 * self.hidden_size, self.source_vocab_size)
        self.copy_linear = nn.Linear(3 * self.hidden_size, 1)
        self.copy_attn = MultiHeadAttention(self.embedding_size, self.num_heads, self.attn_weight_dropout_ratio)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        self.max_target_length = config['max_seq_length']

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mkgraph(self, corpus):
        triples_text = corpus['source_triple']
        source_entity = corpus['source_entity']
        entity_len_ = [len(entity) for entity in source_entity]
        self.batch_size = len(triples_text)
        batch_graph = []
        entity_list = []
        entity_len_list = []
        rel_len = 7
        for i in range(self.batch_size):
            entity_set = source_entity[i]
            entity_len = len(entity_set)
            graph = dgl.DGLGraph()
            # entity node
            graph.add_nodes(entity_len, {'type': torch.ones(entity_len) * self.NODE_TYPE['entity']})
            # root node
            graph.add_nodes(1, {'type': torch.ones(1) * self.NODE_TYPE['root']})
            # relation node
            graph.add_nodes(rel_len * 2, {'type': torch.ones(rel_len * 2) * self.NODE_TYPE['relation']})
            # edge from root to every entity
            graph.add_edges(entity_len, torch.arange(entity_len))
            # edge from every entity to root
            graph.add_edges(torch.arange(entity_len), entity_len)
            # edge from every relation to itself
            graph.add_edges(torch.arange(entity_len + 1 + 2 * rel_len),
                            torch.arange(entity_len + 1 + 2 * rel_len))
            adjust_edge = []
            for st, rel, ed in triples_text[i]:
                st_index, rel_index, ed_index = entity_set.index(' '.join(st)), self.source_relation_token2idx[
                    ''.join(rel)], entity_set.index(' '.join(ed))
                adjust_edge.append([entity_len + rel_index * 2 + 1, st_index])
                adjust_edge.append([ed_index, entity_len + 1 + 2 * rel_index])
                adjust_edge.append([st_index, entity_len + rel_index * 2 + 2])
                adjust_edge.append([entity_len + rel_index * 2 + 2, ed_index])

            if len(adjust_edge) != 0:
                graph.add_edges(*list(map(list, zip(*adjust_edge))))
            batch_graph.append(graph)
            for j in entity_set:
                if "" in j.split(" "):
                    print(entity_set)
            entity_idx_set = [tuple(j.split(" ")) for j in entity_set]
            entity_len_set = [len(j.split(" ")) for j in entity_set]
            entity_idx_set = [[self.source_entity_token2idx[entity] for entity in j] for j in entity_idx_set]
            entity_list.extend(entity_idx_set)
            entity_len_list.extend(entity_len_set)
        graph = dgl.batch(batch_graph)
        return entity_list, entity_len_list, entity_len_, graph

    def encoder(self, corpus):
        # source_text (torch.Tensor): shape: [batch_size, max_seq_len].
        source_idx = corpus['source_idx']
        # source_length (torch.Tensor): shape: [batch_size].
        source_length = corpus['source_length']

        # entity encoder
        # entity_len_list: Length of each entity
        # entity_len: Number of entities in each batch data
        entity_list, entity_len_list, entity_len, graph = self.mkgraph(corpus)
        entity_list, _, _ = pad_sequence(entity_list, entity_len_list, self.padding_token_idx)
        entity_list = entity_list.to(self.device)

        entity_len_list = torch.tensor(entity_len_list).to(self.device)
        entity_len = torch.tensor(entity_len).to(self.device)
        _, [entity_embeddings, c0] = self.entity_encoder(
            self.source_token_embedder(entity_list), entity_len_list
        )
        entity_embeddings = entity_embeddings.transpose(0, 1).contiguous()
        entity_embeddings = entity_embeddings[:, -2:].view(entity_embeddings.size(0), -1)

        # relation encoder
        rel_embeddings = self.rel_token_embedder(torch.arange(15).to(self.device))

        # title encoder
        source_embeddings, _ = self.source_encoder(
            self.source_token_embedder(source_idx), source_length
        )

        # graph encoder
        entity_embeddings, root_embeddings = self.graph_encoder(
            entity_embeddings, entity_len, rel_embeddings, graph, self.device
        )
        return entity_embeddings, source_embeddings, root_embeddings, entity_len

    def generate(self, batch_data, eval_data):
        generate_corpus = []
        idx2token = eval_data.idx2token
        source_idx = batch_data['source_idx']
        self.batch_size = source_idx.size(0)

        entity_embeddings, source_embeddings, root_embeddings, entity_len = self.encoder(batch_data)
        root_embeddings = root_embeddings.unsqueeze(0)

        c = root_embeddings.clone().detach()

        encoder_title_masks = torch.eq(source_idx, self.padding_token_idx).to(self.device)
        encoder_entity_masks = [torch.cat([torch.zeros(i), torch.ones(entity_embeddings.size(1) - i)]).unsqueeze(0)
                                for i in entity_len]
        encoder_entity_masks = torch.cat(encoder_entity_masks, dim=0).bool().to(self.device)
        encoder_states = (root_embeddings, c)

        for bid in range(self.source_text.size(0)):
            decoder_states = encoder_states[bid, :, :]
            entity_embeddings_ = entity_embeddings[bid, :, :]
            source_embeddings_ = source_embeddings[bid, :, :]
            encoder_title_masks_ = encoder_title_masks[bid, :, :]
            encoder_entity_masks_ = encoder_entity_masks[bid, :, :]
            generate_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

            if self.strategy == 'beam_search':
                hypothesis = Beam_Search_Hypothesis(
                    self.beam_size, self.sos_token_idx, self.eos_token_idx, self.device, idx2token
                )

            for gen_idx in range(self.max_target_length):
                decoder_input = self.target_token_embedder(input_seq)
                decoder_outputs, decoder_states = self.decoder(
                    decoder_input, decoder_states, entity_embeddings_, source_embeddings_, encoder_entity_masks_,
                    encoder_title_masks_
                )

                token_logits = self.vocab_linear(decoder_outputs)
                if self.strategy == 'topk_sampling':
                    token_idx = topk_sampling(token_logits).item()
                elif self.strategy == 'greedy_search':
                    token_idx = greedy_search(token_logits).item()
                elif self.strategy == 'beam_search':
                    input_seq, decoder_states, encoder_output = \
                        hypothesis.step(gen_idx, token_logits, decoder_states, encoder_output)

                if self.strategy in ['topk_sampling', 'greedy_search']:
                    if token_idx == self.eos_token_idx:
                        break
                    else:
                        generate_tokens.append(idx2token[token_idx])
                        input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                elif self.strategy == 'beam_search':
                    if (hypothesis.stop()):
                        break

            if self.strategy == 'beam_search':
                generate_tokens = hypothesis.generate()

            generate_corpus.append(generate_tokens)

        return generate_corpus

    def forward(self, corpus, epoch_idx=0):
        # target_idx (torch.Tensor): shape: [batch_size].
        target_idx = corpus['target_idx']
        # target_length (Torch.Tensor): shape: [batch_size]
        target_length = corpus['target_length']
        # source_idx (Torch.Tensor): shape: [batch_size]
        source_idx = corpus['source_idx']

        input_text = target_idx[:, :-1]
        target_text = target_idx[:, 1:]
        entity_embeddings, source_embeddings, root_embeddings, entity_len = self.encoder(corpus)
        root_embeddings = root_embeddings.unsqueeze(0)

        input_embeddings = self.dropout(self.target_token_embedder(input_text))
        c = root_embeddings.clone().detach()

        encoder_title_masks = torch.eq(source_idx, self.padding_token_idx).to(self.device)
        encoder_entity_masks = [torch.cat([torch.zeros(i), torch.ones(entity_embeddings.size(1) - i)]).unsqueeze(0)
                                for i in entity_len]
        encoder_entity_masks = torch.cat(encoder_entity_masks, dim=0).bool().to(self.device)
        decoder_outputs, decoder_states = self.decoder(
            input_embeddings, (root_embeddings, c), entity_embeddings, source_embeddings, encoder_entity_masks,
            encoder_title_masks
        )

        # copy_prob = torch.sigmoid(self.copy_linear(decoder_outputs))
        # vocab_pred = (1 - copy_prob) * self.vocab_linear(decoder_outputs)
        # copy_pred = copy_prob * self.copy_attn(decoder_outputs, entity_embeddings)

        # token_logits (Torch.Tensor): shape: [batch_size, target_length, vocabulary_size].
        token_logits = self.vocab_linear(decoder_outputs)

        # target_text.reshape(-1) (Torch.Tensor): shape: [batch_size * target_length].
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.reshape(-1))
        loss = loss.reshape_as(target_text)
        loss = loss.sum(dim=1) / (target_length - 1).float()
        loss = loss.mean()
        return loss
