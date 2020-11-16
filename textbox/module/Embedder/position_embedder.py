# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


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
