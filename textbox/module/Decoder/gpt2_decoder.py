import torch
from torch import nn
from torch.nn import Parameter
from textbox.module.layers import GPT2TransformerLayer
import torch.nn.functional as F


class GPT2Decoder(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 ffn_size,
                 num_dec_layers,
                 num_heads,
                 attn_dropout_ratio=0.0,
                 attn_weight_dropout_ratio=0.0,
                 ffn_dropout_ratio=0.0,
                 ffn_activate_func='gelu'):
        super(GPT2Decoder, self).__init__()

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_dec_layers):
            self.transformer_layers.append(
                GPT2TransformerLayer(embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio,
                                     ffn_dropout_ratio, ffn_activate_func))

    def forward(self, x, kv=None, self_padding_mask=None, self_attn_mask=None):
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask)
        return x




