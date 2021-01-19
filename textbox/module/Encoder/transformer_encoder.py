# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

r"""
Transformer Encoder
####################
"""

import torch
from torch import nn
from torch.nn import Parameter
from textbox.module.layers import TransformerLayer
import torch.nn.functional as F


class TransformerEncoder(torch.nn.Module):
    r"""
    The stacked Transformer encoder layers.
    """

    def __init__(
        self,
        embedding_size,
        ffn_size,
        num_enc_layers,
        num_heads,
        attn_dropout_ratio=0.0,
        attn_weight_dropout_ratio=0.0,
        ffn_dropout_ratio=0.0
    ):
        super(TransformerEncoder, self).__init__()

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_enc_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio,
                    ffn_dropout_ratio
                )
            )

    def forward(self, x, kv=None, self_padding_mask=None, output_all_encoded_layers=False):
        r""" Implement the encoding process step by step.

        Args:
            x (Torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            kv (Torch.Tensor): the cached history latent vector, shape: [batch_size, sequence_length, embedding_size], default: None.
            self_padding_mask (Torch.Tensor): padding mask of target sequence, shape: [batch_size, sequence_length], default: None.
            output_all_encoded_layers (Bool): whether to output all the encoder layers, default: ``False``.

        Returns:
            Torch.Tensor: output features, shape: [batch_size, sequence_length, ffn_size].
        """
        all_encoded_layers = []
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask)
            all_encoded_layers.append(x)
        if output_all_encoded_layers:
            return all_encoded_layers
        return all_encoded_layers[-1]
