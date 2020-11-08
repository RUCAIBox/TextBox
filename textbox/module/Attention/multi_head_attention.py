import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.0, attn_weights_dropout=True):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_size = embedding_size // num_heads

        assert self.head_size * num_heads == self.embedding_size, "embedding size must be divisible by num_heads"

        self.scaling = self.head_size ** -0.5  # d_k ** -0.5

        self.query_proj = nn.Linear(embedding_size, embedding_size)
        self.key_proj = nn.Linear(embedding_size, embedding_size)
        self.value_proj = nn.Linear(embedding_size, embedding_size)

        self.out_proj = nn.Linear(embedding_size, embedding_size)
        self.attn_weights_dropout = attn_weights_dropout

        self.dropout_layer = nn.Dropout(self.dropout)

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

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.attn_weights_dropout:
            attn_weights = self.dropout_layer(attn_weights)

        attn_repre = torch.matmul(attn_weights, v)  # [batch_size, num_heads, tgt_len, head_size]

        if not self.attn_weights_dropout:
            attn_repre = self.dropout_layer(attn_repre)

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


class LearnedPositionalEmbedding(nn.Module):
    """This module produces LearnedPositionalEmbedding.
    """
    def __init__(self, embedding_size, max_length=512):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(max_length, embedding_size)

    def forward(self, input_seq, offset=0):
        """Input is expected to be of size [batch_size x seq_len]."""
        batch_size, seq_len = input_seq.size()
        positions = (offset + torch.arange(seq_len))
        pos_embeddings = self.weights(positions).unsqueeze(0).expand(batch_size, -1, -1)
        return pos_embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    """
    def __init__(self, embedding_size, max_length=512):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            max_length,
            embedding_size
        )

    @staticmethod
    def get_embedding(max_length, embedding_size):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_length, -1)
        if embedding_size % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
        return emb

    def forward(self, input_seq, offset=0):
        """Input is expected to be of size [batch_size x seq_len]."""
        batch_size, seq_len = input_seq.size()
        max_position = seq_len + offset
        if self.weights is None or max_position > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_position,
                self.embedding_size,
            )

        positions = offset + torch.arange(seq_len)
        pos_embeddings = self.weights.index_select(0, positions).unsqueeze(0).expand(batch_size, -1, -1).detach()
        return pos_embeddings
