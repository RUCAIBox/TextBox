import torch
from torch import nn
from torch.nn import Parameter
from textbox.module.Attention.multi_head_attention import MultiHeadAttention
import torch.nn.functional as F


class TransformerDecoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 ffn_size,
                 num_layers,
                 num_heads,
                 dropout=0.0,
                 with_external=False,
                 attn_weights_dropout=True):
        super(TransformerDecoder, self).__init__()
        
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerLayer(input_size, ffn_size, num_heads, dropout, with_external, attn_weights_dropout))

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_memories=None, external_padding_mask=None):
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask, external_memories, external_padding_mask)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, ffn_size, num_heads, dropout=0.0,
                 with_external=False, attn_weights_dropout=True):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_size, num_heads, dropout, attn_weights_dropout)
        self.ffn_1 = nn.Linear(embedding_size, ffn_size)
        self.ffn_2 = nn.Linear(ffn_size, embedding_size)

        self.attn_layer_norm = nn.LayerNorm(embedding_size)
        self.ffn_layer_norm = nn.LayerNorm(embedding_size)

        self.with_external = with_external
        self.dropout_layer = nn.Dropout(dropout)

        if self.with_external:
            self.external_attention = MultiHeadAttention(embedding_size, num_heads, dropout, attn_weights_dropout)
            self.external_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_memories=None, external_padding_mask=None):
        # x: batch_size x seq_len x embed_dim
        residual = x
        if kv is None:
            x, self_attn_weights = self.self_attention(query=x, key=x, value=x,
                                                       key_padding_mask=self_padding_mask,
                                                       attn_mask=self_attn_mask)
        else:
            x, self_attn_weights = self.self_attention(query=x, key=kv, value=kv,
                                                       key_padding_mask=self_padding_mask,
                                                       attn_mask=self_attn_mask)
        x = self.dropout_layer(x)
        x = self.attn_layer_norm(residual + x)  # LayerNorm(residual + Sublayer(x))

        if self.with_external:
            # attention on encoder states
            residual = x
            x, external_attn_weights = self.external_attention(query=x, key=external_memories, value=external_memories,
                                                               key_padding_mask=external_padding_mask)
            x = self.dropout_layer(x)
            x = self.external_layer_norm(residual + x)  # LayerNorm(residual + Sublayer(x))
        else:
            external_attn_weights = None

        residual = x
        x = F.gelu(self.ffn_1(x))  # nn.ReLu
        x = self.dropout_layer(x)
        x = self.ffn_2(x)
        x = self.dropout_layer(x)
        x = self.ffn_layer_norm(residual + x)  # LayerNorm(residual + Sublayer(x))

        return x, self_attn_weights, external_attn_weights



