import torch
from torch import nn
from torch.nn import Parameter
from textbox.module.layers import TransformerLayer
import torch.nn.functional as F


class TransformerDecoder(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 ffn_size,
                 num_dec_layers,
                 num_heads,
                 attn_dropout_ratio=0.0,
                 attn_weight_dropout_ratio=0.0,
                 ffn_dropout_ratio=0.0,
                 ffn_activate_func='gelu',
                 with_external=True):
        super(TransformerDecoder, self).__init__()
        
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_dec_layers):
            self.transformer_layers.append(
                TransformerLayer(embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio,
                                 ffn_dropout_ratio, ffn_activate_func, with_external))

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_states=None, external_padding_mask=None):
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask, external_states, external_padding_mask)
        return x




