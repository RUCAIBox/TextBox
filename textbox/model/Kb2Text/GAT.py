import math
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from torch import nn
import torch

from textbox.data.utils import pad_sequence

NODE_TYPE = {'entity': 0, 'root': 1, 'relation': 2}


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
        self.query_proj = nn.Linear(embedding_size, embedding_size)
        self.key_proj = nn.Linear(embedding_size, embedding_size)
        self.value_proj = nn.Linear(embedding_size, embedding_size)
        self.attn_drop = nn.Dropout(attn_drop)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.out_proj = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.PReLU(4 * embedding_size),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(ffn_drop),
        )
        # a strange FFN

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
        # use the same layer norm, see the author's code
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
