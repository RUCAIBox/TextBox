# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

r"""
Transformer Decoder
###################
"""

import torch
from torch import nn
from torch.nn import Parameter
from textbox.module.layers import TransformerLayer
import torch.nn.functional as F


class TransformerDecoder(torch.nn.Module):
    r"""
    The stacked Transformer decoder layers.
    """

    def __init__(
        self,
        embedding_size,
        ffn_size,
        num_dec_layers,
        num_heads,
        attn_dropout_ratio=0.0,
        attn_weight_dropout_ratio=0.0,
        ffn_dropout_ratio=0.0,
        with_external=True
    ):
        super(TransformerDecoder, self).__init__()

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_dec_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio,
                    ffn_dropout_ratio, with_external
                )
            )

    def forward(
        self,
        x,
        kv=None,
        self_padding_mask=None,
        self_attn_mask=None,
        external_states=None,
        external_padding_mask=None
    ):
        r""" Implement the decoding process step by step.

        Args:
            x (Torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            kv (Torch.Tensor): the cached history latent vector, shape: [batch_size, sequence_length, embedding_size], default: None.
            self_padding_mask (Torch.Tensor): padding mask of target sequence, shape: [batch_size, sequence_length], default: None.
            self_attn_mask (Torch.Tensor): diagonal attention mask matrix of target sequence, shape: [batch_size, sequence_length, sequence_length], default: None.
            external_states (Torch.Tensor): output features of encoder, shape: [batch_size, sequence_length, feature_size], default: None.
            external_padding_mask (Torch.Tensor): padding mask of source sequence, shape: [batch_size, sequence_length], default: None.

        Returns:
            Torch.Tensor: output features, shape: [batch_size, sequence_length, ffn_size].
        """
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask, external_states, external_padding_mask)
        return x
