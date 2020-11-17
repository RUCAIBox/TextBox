# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class LuongAttention(torch.nn.Module):
    def __init__(self):
        pass


class BahdanauAttention(torch.nn.Module):
    def __init__(self):
        pass


class MonotonicAttention(torch.nn.Module):
    def __init__(self):
        pass


class LuongMonotonicAttention(torch.nn.Module):
    def __init__(self):
        pass


class BahdanauMonotonicAttention(torch.nn.Module):
    def __init__(self):
        pass


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_size, num_heads, attn_weights_dropout_ratio=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        assert self.head_size * num_heads == self.embedding_size, "embedding size must be divisible by num_heads"

        self.scaling = self.head_size ** -0.5  # d_k ** -0.5

        self.query_proj = nn.Linear(embedding_size, embedding_size)
        self.key_proj = nn.Linear(embedding_size, embedding_size)
        self.value_proj = nn.Linear(embedding_size, embedding_size)

        self.out_proj = nn.Linear(embedding_size, embedding_size)

        self.weight_dropout = nn.Dropout(attn_weights_dropout_ratio)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """ Input shape: batch_size * time * embedding_size
            key_padding_mask: batch_size * time
            attention_mask:  tgt_len x src_len
        """
        batch_size, tgt_len, embedding_size = query.size()
        src_len = key.size(1)
        assert key.size() == value.size()

        q = self.query_proj(query) * self.scaling
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(batch_size, tgt_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, src_len, self.num_heads, self.head_size).permute(0, 2, 3, 1)
        v = v.view(batch_size, src_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k)
        assert list(attn_weights.size()) == [batch_size, self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            # don't attend to future symbols
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),
                1e-10
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                1e-10
            )

        attn_weights = self.weight_dropout(F.softmax(attn_weights, dim=-1))
        attn_repre = torch.matmul(attn_weights, v)  # [batch_size, num_heads, tgt_len, head_size]

        assert list(attn_repre.size()) == [batch_size, self.num_heads, tgt_len, self.head_size]

        attn_repre = attn_repre.transpose(1, 2).contiguous().view(batch_size, tgt_len, embedding_size)
        attn_repre = self.out_proj(attn_repre)

        # maximum attention weight over heads
        attn_weights, _ = attn_weights.max(dim=1)

        return attn_repre, attn_weights


class SelfAttentionMask(torch.nn.Module):
    def __init__(self, init_size=100):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)

    @staticmethod
    def get_mask(size):
        weights = torch.ones((size, size), dtype=torch.uint8).triu_(1)  # above the diagonal == 1
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        masks = self.weights[:size, :size].detach()
        return masks


