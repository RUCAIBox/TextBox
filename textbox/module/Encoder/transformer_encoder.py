import torch
from torch import nn
from torch.nn import Parameter
from textbox.module.layers import TransformerLayer
import torch.nn.functional as F


class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 ffn_size,
                 num_enc_layers,
                 num_heads,
                 attn_dropout_ratio=0.0,
                 attn_weight_dropout_ratio=0.0,
                 ffn_dropout_ratio=0.0):
        super(TransformerEncoder, self).__init__()

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_enc_layers):
            self.transformer_layers.append(
                TransformerLayer(embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio,
                                 ffn_dropout_ratio))

    def forward(self, x, kv=None, self_padding_mask=None, output_all_encoded_layers=False):
        all_encoded_layers = []
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask)
            all_encoded_layers.append(x)
        if output_all_encoded_layers:
            return all_encoded_layers
        return all_encoded_layers[-1]




