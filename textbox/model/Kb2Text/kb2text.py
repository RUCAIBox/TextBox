# @Time   : 2021/12/3
# @Author : ZiKang Liu
# @Email  ：jason8121@foxmail.com

r"""
Graphwriter
################################################
Reference:
    Rik et al. "Text Generation from Knowledge Graphs with Graph Transformers" in ACL 2019.
"""
import math
import re

import dgl
import dgl.function as fn
import torch
from dgl.nn.pytorch import edge_softmax
from torch import nn

from textbox.data.utils import pad_sequence
from textbox.model.abstract_generator import Seq2SeqGenerator
from textbox.model.init import xavier_normal_initialization
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.strategy import topk_sampling, greedy_search, Beam_Search_Hypothesis
from textbox.module.Attention.attention_mechanism import MultiHeadAttention, LuongAttention

NODE_TYPE = {'entity': 0, 'root': 1, 'relation': 2}


class ContextAttentionalDecoder(torch.nn.Module):
    r"""
    Attention-based Recurrent Neural Network (RNN) decoder.
    """

    def __init__(
            self,
            embedding_size,
            hidden_size,
            context_size,
            num_dec_layers,
            rnn_type,
            dropout_ratio=0.0,
            attention_type='LuongAttention',
            alignment_method='concat',
            num_heads=4,
            attn_weight_dropout_ratio=0.1
    ):
        super(ContextAttentionalDecoder, self).__init__()

        self.attn_weight_dropout_ratio = attn_weight_dropout_ratio
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_type = rnn_type
        self.attention_type = attention_type
        self.alignment_method = alignment_method

        dec_input_size = embedding_size * 3
        self.decoder = nn.LSTM(dec_input_size, self.hidden_size, self.num_dec_layers, batch_first=True)
        self.attention_dense = nn.Linear(hidden_size + 2 * context_size, hidden_size)
        self.attentioner = LuongAttention(self.context_size, self.hidden_size, self.alignment_method)
        self.item_attentioner = MultiHeadAttention(embedding_size, self.num_heads, self.attn_weight_dropout_ratio)
        self.title_attentioner = MultiHeadAttention(embedding_size, self.num_heads, self.attn_weight_dropout_ratio)

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape:
            [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        h_0 = torch.zeros(self.num_dec_layers, batch_size, 2 * self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_dec_layers, batch_size, 2 * self.hidden_size).to(device)
        hidden_states = (h_0, c_0)
        return hidden_states

    def forward(
            self, input_embeddings, hidden_states=None, encoder_outputs_items=None, encoder_outputs_titles=None,
            encoder_item_masks=None, encoder_title_masks=None
    ):
        r""" Implement the attention-based decoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.
            encoder_outputs_items: (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_outputs_titles: (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_item_masks (Torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.
            encoder_title_masks (Torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        decode_length = input_embeddings.size(1)

        all_outputs = []
        h_ = hidden_states[0].transpose(0, 1)
        context_items = h_ + self.item_attentioner(
            h_, encoder_outputs_items, encoder_outputs_items, encoder_item_masks)[0]
        context_titles = h_ + self.title_attentioner(
            h_, encoder_outputs_titles, encoder_outputs_titles, encoder_title_masks)[0]
        context = torch.cat((context_items, context_titles), dim=2)

        for step in range(decode_length):
            inputs = input_embeddings[:, step, :].unsqueeze(1)
            inputs = torch.cat([context, inputs], dim=2)
            outputs, hidden_states = self.decoder(inputs, hidden_states)

            h_ = hidden_states[0].transpose(0, 1)
            context_items = h_ + self.item_attentioner(
                h_, encoder_outputs_items, encoder_outputs_items, encoder_item_masks)[0]
            context_titles = h_ + self.title_attentioner(
                h_, encoder_outputs_titles, encoder_outputs_titles, encoder_title_masks)[0]
            context = torch.cat((context_items, context_titles), dim=2)
            outputs = torch.cat((outputs.squeeze(1), context.squeeze(1)), dim=1)
            all_outputs.append(outputs)

        outputs = torch.stack(all_outputs, dim=1)
        return outputs, hidden_states


class GAT(nn.Module):
    # a graph attention network with dot-product attention
    def __init__(self,
                 embedding_size,
                 num_heads,
                 ffn_drop=0.,
                 attn_drop=0.
                 ):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.query_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.key_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.out_proj = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.PReLU(4 * embedding_size),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(ffn_drop),
        )

    def forward(self, graph, feat, device):
        graph = graph.to(device).local_var()
        feat_c = feat.clone().detach().requires_grad_(False)
        q, k, v = self.query_proj(feat), self.key_proj(feat_c), self.value_proj(feat_c)
        q = q.view(-1, self.num_heads, self.embedding_size // self.num_heads)
        k = k.view(-1, self.num_heads, self.embedding_size // self.num_heads)
        v = v.view(-1, self.num_heads, self.embedding_size // self.num_heads)
        graph.ndata.update(
            {'ft': v, 'el': k, 'er': q})  # k,q instead of q,k, the edge_softmax is applied on incoming edges
        # compute edge attention
        graph.apply_edges(fn.u_dot_v('el', 'er', 'e'))
        e = graph.edata.pop('e') / math.sqrt(self.embedding_size)
        graph.edata['a'] = edge_softmax(graph, e)
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft2'))
        rst = graph.ndata['ft2']
        # residual
        rst = rst.view(feat.shape) + feat
        rst = self.ln1(rst)
        rst = self.ln1(rst + self.out_proj(rst))
        return rst


class GraphTransformer(nn.Module):
    def __init__(self, embedding_size, attn_drop, ffn_drop, layer_nums):
        super().__init__()
        self.embedding_size = embedding_size
        self.GAT = nn.ModuleList(
            [GAT(self.embedding_size, 4, attn_drop=attn_drop, ffn_drop=ffn_drop) for _ in
             range(layer_nums)])
        self.layer_nums = layer_nums

    def forward(self, entity_embeddings, entity_len_list, rel_embeddings, graphs, device):
        init_h = []
        cur = 0
        for i in range(graphs.batch_size):
            init_h.append(entity_embeddings[cur:cur + entity_len_list[i]])
            init_h.append(rel_embeddings)
            cur = cur + entity_len_list[i]
        init_h = torch.cat(init_h, 0)
        feats = init_h
        for i in range(self.layer_nums):
            feats = self.GAT[i](graphs, feats, device)
        g_root = feats.index_select(0, graphs.filter_nodes(lambda x: x.data['type'] == NODE_TYPE['root']).to(device))
        g_ent = nn.utils.rnn.pad_sequence(
            feats.index_select(0, graphs.filter_nodes(lambda x: x.data['type'] == NODE_TYPE['entity']).to(
                device)).split(list(entity_len_list)), batch_first=True)
        return g_ent, g_root


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
        self.source_relation_idx2token = dataset.source_relation_idx2token
        self.source_relation_token2idx = dataset.source_relation_token2idx
        self.REL_SET = []
        self.type_vocab = []
        with open(config["relation_vocab"]) as f:
            for line in f:
                self.REL_SET.append(line.rstrip())
        with open(config["type_vocab"]) as f:
            for line in f:
                self.type_vocab.append(line.rstrip())

        self.REL_LEN = 2 * len(self.REL_SET) + 1

        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx
        self.source_token2idx = dataset.source_token2idx
        self.source_entity_token2idx = dataset.source_entity_token2idx
        self.entity_vocab_size = len(self.source_entity_token2idx)

        self.source_token_embedder = nn.Embedding(
            self.source_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.target_token_embedder = nn.Embedding(
            self.target_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.entity_token_embedder = nn.Embedding(
            self.entity_vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        self.rel_token_embedder = nn.Embedding(
            self.REL_LEN, self.embedding_size, padding_idx=self.padding_token_idx
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

        self.decoder = ContextAttentionalDecoder(
            self.embedding_size, self.hidden_size, self.embedding_size, self.num_dec_layers, self.rnn_type,
            self.dropout_ratio, self.attention_type, self.alignment_method
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(3 * self.hidden_size, self.target_vocab_size)
        self.copy_linear = nn.Linear(3 * self.hidden_size, 1)
        self.d_linear = nn.Linear(3 * self.embedding_size, self.embedding_size)
        self.copy_attn = MultiHeadAttention(self.embedding_size, 1, 0., )

        self.loss = nn.NLLLoss(ignore_index=self.padding_token_idx, reduction='none')

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
        # total nums of relations
        rel_len = len(self.REL_SET)
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

        """ entity encoder
        entity_len_list: Length of each entity
        entity_len: Number of entities in each batch data
        """
        entity_list, entity_len_list, entity_len, graph = self.mkgraph(corpus)
        entity_list, _, _ = pad_sequence(entity_list, entity_len_list, self.padding_token_idx)
        entity_list = entity_list.to(self.device)

        entity_len_list = torch.tensor(entity_len_list).to(self.device)
        entity_len = torch.tensor(entity_len).to(self.device)
        _, [entity_embeddings, c0] = self.entity_encoder(
            self.entity_token_embedder(entity_list), entity_len_list
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
        idx2token = eval_data.dataset.target_idx2token
        source_idx = batch_data['source_idx']
        source_entity = batch_data['source_entity']
        target_dict = batch_data['target_dict']
        self.batch_size = source_idx.size(0)

        entity_embeddings, source_embeddings, root_embeddings, entity_len = self.encoder(batch_data)
        root_embeddings = root_embeddings.unsqueeze(0)

        c = root_embeddings.clone().detach()

        encoder_title_masks = torch.eq(source_idx, self.padding_token_idx).to(self.device)
        encoder_entity_masks = [torch.cat([torch.zeros(i), torch.ones(entity_embeddings.size(1) - i)]).unsqueeze(0)
                                for i in entity_len]
        encoder_entity_masks = torch.cat(encoder_entity_masks, dim=0).bool().to(self.device)

        for bid in range(self.batch_size):
            decoder_states = (root_embeddings[:, bid, :].unsqueeze(1), c[:, bid, :].unsqueeze(1))
            entity_embeddings_ = entity_embeddings[bid, :, :].unsqueeze(0)
            source_embeddings_ = source_embeddings[bid, :, :].unsqueeze(0)
            encoder_title_masks_ = encoder_title_masks[bid, :].unsqueeze(0)
            encoder_entity_masks_ = encoder_entity_masks[bid, :].unsqueeze(0)
            generate_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

            if self.strategy == 'beam_search':
                hypothesis = Beam_Search_Hypothesis(
                    self.beam_size, self.sos_token_idx, self.eos_token_idx, self.device, idx2token
                )

            for gen_idx in range(self.target_max_length):
                decoder_input = self.target_token_embedder(input_seq)
                decoder_outputs, decoder_states = self.decoder(
                    decoder_input, decoder_states, entity_embeddings_, source_embeddings_, encoder_entity_masks_,
                    encoder_title_masks_
                )
                copy_prob = torch.sigmoid(self.copy_linear(decoder_outputs))

                EPSI = torch.tensor(1e-6)
                pred_vocab = torch.log(copy_prob + EPSI) + torch.log_softmax(self.vocab_linear(decoder_outputs), -1)
                douts = self.d_linear(decoder_outputs)
                attn_weight = self.copy_attn(douts, entity_embeddings_, entity_embeddings_, encoder_entity_masks_)[2]
                pred_copy = torch.log((1. - copy_prob) + EPSI) + attn_weight.squeeze(1)
                token_logits = torch.cat([pred_vocab, pred_copy], -1)

                # we only support greedy search for this task
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
                    elif token_idx >= self.target_vocab_size:
                        entity_tokens = source_entity[bid][token_idx - self.target_vocab_size]
                        entity_tokens = entity_tokens.split(" ")
                        generate_tokens.extend(entity_tokens)
                        # retrieve next token
                        next_token = self.target_token2idx[target_dict[bid][token_idx - self.target_vocab_size]]
                        input_seq = torch.LongTensor([[next_token]]).to(self.device)
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
        mention_text = corpus['target_mention']

        mention_idx = target_text.cpu().numpy()
        # entity_tag = ["<task>", "<material>", "<method>", "<otherscientificterm>", "<metric>"]
        entity_tag = [self.target_token2idx[i] for i in self.type_vocab]
        for sent_idx in range(self.batch_size):
            for j in range(mention_idx.shape[1]):
                if mention_idx[sent_idx][j] in entity_tag:
                    entity_num = re.findall("\d+", mention_text[sent_idx][j])[0]
                    mention_idx[sent_idx][j] = self.target_vocab_size + int(entity_num)

        mention_idx = torch.from_numpy(mention_idx).to(self.device)

        entity_embeddings, source_embeddings, root_embeddings, entity_len = self.encoder(corpus)
        root_embeddings = root_embeddings.unsqueeze(0)

        input_embeddings = self.target_token_embedder(input_text)
        c = root_embeddings.clone().detach()

        encoder_title_masks = torch.eq(source_idx, self.padding_token_idx).to(self.device)
        encoder_entity_masks = [torch.cat([torch.zeros(i), torch.ones(entity_embeddings.size(1) - i)]).unsqueeze(0)
                                for i in entity_len]
        encoder_entity_masks = torch.cat(encoder_entity_masks, dim=0).bool().to(self.device)
        decoder_outputs, decoder_states = self.decoder(
            input_embeddings, (root_embeddings, c), entity_embeddings, source_embeddings, encoder_entity_masks,
            encoder_title_masks
        )

        copy_prob = torch.sigmoid(self.copy_linear(decoder_outputs))

        EPSI = torch.tensor(1e-6)
        pred_vocab = torch.log(copy_prob + EPSI) + torch.log_softmax(self.vocab_linear(decoder_outputs), -1)
        douts = self.d_linear(decoder_outputs)
        attn_weight = self.copy_attn(douts, entity_embeddings, entity_embeddings, encoder_entity_masks)
        pred_copy = torch.log((1. - copy_prob) + EPSI) + attn_weight[2].squeeze(1)
        token_logits = torch.cat([pred_vocab, pred_copy], -1)

        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), mention_idx.reshape(-1))
        loss = loss.reshape_as(mention_idx)
        loss = loss.sum(dim=1) / (target_length - 1).float()
        loss = loss.mean()
        return loss
