import torch
from torch import nn
from torch.nn import Parameter
from textbox.module.layers import TransformerLayer
import torch.nn.functional as F


class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 ffn_size,
                 num_layers,
                 num_heads,
                 attn_dropout=0.0,
                 attn_weight_dropout=0.0,
                 ffn_dropout=0.0,
                 ffn_activate_func='gelu'):
        super(TransformerEncoder, self).__init__()

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerLayer(input_size, ffn_size, num_heads, attn_dropout, attn_weight_dropout,
                                 ffn_dropout, ffn_activate_func))

    def forward(self, x, kv=None, self_padding_mask=None, output_all_encoded_layers=True):
        all_encoded_layers = []
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask)
            all_encoded_layers.append(x)
        if not output_all_encoded_layers:
            return all_encoded_layers[-1]
        return all_encoded_layers




