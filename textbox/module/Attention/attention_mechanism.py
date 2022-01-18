# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/26
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn

r"""
Attention Layers
################################################
"""

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class LuongAttention(torch.nn.Module):
    r"""Luong Attention is proposed in the following paper: Effective Approaches to Attention-based Neural Machine Translation.

    Reference:
        https://arxiv.org/abs/1508.04025
    """

    def __init__(self, source_size, target_size, alignment_method='concat', is_coverage=False):
        super(LuongAttention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size
        self.alignment_method = alignment_method

        self.is_coverage = is_coverage
        if self.is_coverage:
            self.coverage_linear = nn.Linear(1, target_size, bias=False)

        if self.alignment_method == 'general':
            self.energy_linear = nn.Linear(target_size, source_size, bias=False)
        elif self.alignment_method == 'concat':
            self.energy_linear = nn.Linear(source_size + target_size, target_size)
            self.v = nn.Parameter(torch.rand(target_size, dtype=torch.float32))
        elif self.alignment_method == 'dot':
            assert self.source_size == target_size
        else:
            raise ValueError("The alignment method for Luong Attention must be in ['general', 'concat', 'dot'].")

    def score(self, hidden_states, encoder_outputs, coverages=None):
        r"""Calculate the attention scores between encoder outputs and decoder states."""
        tgt_len = hidden_states.size(1)
        src_len = encoder_outputs.size(1)

        if self.alignment_method == 'general':
            energy = self.energy_linear(hidden_states)
            encoder_outputs = encoder_outputs.permute(0, 2, 1)
            energy = energy.bmm(encoder_outputs)
            return energy
        elif self.alignment_method == 'concat':
            hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, src_len, 1)  # B * tgt_len * src_len * target_size
            encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, tgt_len, 1, 1)
            energy = self.energy_linear(torch.cat((hidden_states, encoder_outputs), dim=-1))
            if self.is_coverage:
                coverages = self.coverage_linear(coverages.unsqueeze(3))
                energy = energy + coverages
            energy = torch.tanh(energy)
            energy = self.v.mul(energy).sum(dim=-1)
            return energy
        elif self.alignment_method == 'dot':
            encoder_outputs = encoder_outputs.permute(0, 2, 1)
            energy = hidden_states.bmm(encoder_outputs)
            return energy
        else:
            raise NotImplementedError(
                "No such alignment method {} for computing Luong scores.".format(self.alignment_method)
            )

    def forward(self, hidden_states, encoder_outputs, encoder_masks, coverages=None):
        r"""
        Luong attention

        Args:
            hidden_states: shape: [batch_size, tgt_len, target_size]
            encoder_outputs: shape: [batch_size, src_len, source_size]
            encoder_masks: shape: [batch_size, src_len]

        Return:
            tuple:
                - context: shape: [batch_size, tgt_len, source_size]
                - probs: shape: [batch_size, tgt_len, src_len]
        """
        tgt_len = hidden_states.size(1)
        energy = self.score(hidden_states, encoder_outputs, coverages=coverages)
        probs = F.softmax(energy, dim=-1) * encoder_masks.unsqueeze(1).repeat(1, tgt_len, 1)
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        context = probs.bmm(encoder_outputs)

        if self.is_coverage:
            coverages = probs + coverages
            return context, probs, coverages

        return context, probs


class BahdanauAttention(torch.nn.Module):
    r"""Bahdanau Attention is proposed in the following paper:
            Neural Machine Translation by Jointly Learning to Align and Translate.

    Reference:
        https://arxiv.org/abs/1409.0473
    """

    def __init__(self, source_size, target_size):
        super(BahdanauAttention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size

        self.energy_linear = nn.Linear(source_size + target_size, target_size)
        self.v = nn.Parameter(torch.FloatTensor(target_size))

    def score(self, hidden_states, encoder_outputs):
        r"""Calculate the attention scores between encoder outputs and decoder states."""
        src_len = encoder_outputs.size(1)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.energy_linear(torch.cat((hidden_states, encoder_outputs), dim=-1)))
        energy = self.v.mul(energy).sum(dim=-1)
        return energy

    def forward(self, hidden_states, encoder_outputs, encoder_masks):
        r"""
        Bahdanau attention

        Args:
            hidden_states: shape: [batch_size, tgt_len, target_size]
            encoder_outputs: shape: [batch_size, src_len, source_size]
            encoder_masks: shape: [batch_size, src_len]

        Return:
            tuple:
                - context: shape: [batch_size, tgt_len, source_size]
                - probs: shape: [batch_size, tgt_len, src_len]
        """
        energy = self.score(hidden_states, encoder_outputs)
        probs = F.softmax(energy, dim=-1) * encoder_masks
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        probs = probs.unsqueeze(1)
        context = probs.bmm(encoder_outputs)

        return context, probs


class MonotonicAttention(torch.nn.Module):
    r"""Monotonic Attention is proposed in the following paper:
        Online and Linear-Time Attention by Enforcing Monotonic Alignments.

    Reference:
        https://arxiv.org/abs/1704.00784
    """

    def __init__(self, source_size, target_size, init_r=-4):
        super(MonotonicAttention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size

        self.w_linear = nn.Linear(source_size, target_size)
        self.v_linear = nn.Linear(target_size, target_size)
        self.bias = nn.Parameter(torch.Tensor(target_size).normal_())

        self.v = nn.utils.weight_norm(nn.Linear(target_size, 1))
        self.v.weight_g.data = torch.Tensor([1 / target_size]).sqrt()

        self.r = nn.Parameter(torch.Tensor([init_r]))

    def gaussian_noise(self, *size):
        r"""Additive gaussian nosie to encourage discreteness"""
        return torch.FloatTensor(*size).normal_()

    def safe_cumprod(self, x):
        r"""Numerically stable cumulative product by cumulative sum in log-space"""
        return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1))

    def exclusive_cumprod(self, x):
        r"""Exclusive cumulative product [a, b, c] => [1, a, a * b]"""
        batch_size = x.size(0)
        ones = torch.ones(batch_size, 1).to(x.device)
        one_x = torch.cat((ones, x), dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def score(self, hidden_states, encoder_outputs):
        r"""Calculate the attention scores between encoder outputs and decoder states."""
        tgt_len = hidden_states.size(1)
        src_len = encoder_outputs.size(1)
        energy = torch.tanh(
            self.w_linear(encoder_outputs).unsqueeze(1).repeat(1, tgt_len, 1, 1) +
            self.v_linear(hidden_states).unsqueeze(2).repeat(1, 1, src_len, 1) + self.bias
        )
        energy = self.v(energy).squeeze(-1) + self.r
        return energy

    def soft(self, hidden_states, encoder_outputs, encoder_masks, previous_probs=None):
        r"""
        Soft monotonic attention (Train)

        Args:
            hidden_states: shape: [batch_size, tgt_len, target_size]
            encoder_outputs: shape: [batch_size, src_len, source_size]
            encoder_masks: shape: [batch_size, src_len]
            previous_probs: shape: [batch_size, tgt_len, src_len]

        Return:
            tuple:
                - context: shape: [batch_size, tgt_len, source_size]
                - probs: shape: [batch_size, tgt_len, src_len]
        """
        device = hidden_states.device
        tgt_len = hidden_states.size(1)
        batch_size, src_len, _ = encoder_outputs.size()

        energy = self.score(hidden_states, encoder_outputs)
        p_select = torch.sigmoid(energy + self.gaussian_noise(energy.size()).to(device))
        cumprod_1_minus_p = self.safe_cumprod(1 - p_select)

        if previous_probs is None:
            probs = torch.zeros(batch_size, tgt_len, src_len).to(device)
            probs[:, :, 0] = torch.ones(batch_size, tgt_len).to(device)
        else:
            probs = p_select * cumprod_1_minus_p * torch.cumsum(previous_probs / cumprod_1_minus_p, dim=-1)

        encoder_masks = encoder_masks.unsqueeze(1).repeat(1, tgt_len, 1)
        probs = probs * encoder_masks
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        context = probs.bmm(encoder_outputs)

        return context, probs

    def hard(self, hidden_states, encoder_outputs, encoder_masks, previous_probs=None):
        r"""
        Hard monotonic attention (Test)

        Args:
            hidden_states: shape: [batch_size, tgt_len, target_size]
            encoder_outputs: shape: [batch_size, src_len, source_size]
            encoder_masks: shape: [batch_size, src_len]
            previous_probs: shape: [batch_size, tgt_len, src_len]

        Return:
            tuple:
                - context: shape: [batch_size, tgt_len, source_size]
                - probs: shape: [batch_size, tgt_len, src_len]
        """
        device = hidden_states.device
        tgt_len = hidden_states.size(1)
        batch_size, src_len, _ = encoder_outputs.size()

        if previous_probs is None:
            probs = torch.zeros(batch_size, tgt_len, src_len).to(device)
            probs[:, :, 0] = torch.ones(batch_size, tgt_len).to(device)
        else:
            energy = self.score(hidden_states, encoder_outputs)

            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            above_threshold = (energy > 0).float()

            p_select = above_threshold * torch.cumsum(previous_probs, dim=-1)
            probs = p_select * self.exclusive_cumprod(1 - p_select)

            # Not attended => attend at last encoder output
            # Assume that encoder outputs are not padded
            attended = probs.sum(dim=-1)
            for batch_i in range(batch_size):
                if not attended[batch_i]:
                    probs[batch_i, -1] = 1

        encoder_masks = encoder_masks.unsqueeze(1).repeat(1, tgt_len, 1)
        probs = probs * encoder_masks
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        context = probs.bmm(encoder_outputs)

        return context, probs


class MultiHeadAttention(torch.nn.Module):
    r"""Multi-head Attention is proposed in the following paper:
            Attention Is All You Need.

    Reference:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, embedding_size, num_heads, attn_weight_dropout_ratio=0.0, return_distribute=False):
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

        self.weight_dropout = nn.Dropout(attn_weight_dropout_ratio)
        self.return_distribute = return_distribute

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.query_proj.bias, 0.)
        nn.init.constant_(self.key_proj.bias, 0.)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        r"""
        Multi-head attention

        Args:
            query: shape: [batch_size, tgt_len, embedding_size]
            key and value: shape: [batch_size, src_len, embedding_size]
            key_padding_mask: shape: [batch_size, src_len]
            attn_mask: shape: [batch_size, tgt_len, src_len]

        Return:
            tuple:
                - attn_repre: shape: [batch_size, tgt_len, embedding_size]
                - attn_weights: shape: [batch_size, tgt_len, src_len]
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
            attn_weights.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        if key_padding_mask is not None:
            attn_weights.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights_ = torch.log_softmax(attn_weights, -1)
        attn_weights = self.weight_dropout(F.softmax(attn_weights, dim=-1))
        attn_repre = torch.matmul(attn_weights, v)

        assert list(attn_repre.size()) == [batch_size, self.num_heads, tgt_len, self.head_size]

        attn_repre = attn_repre.transpose(1, 2).contiguous().view(batch_size, tgt_len, embedding_size)
        attn_repre = self.out_proj(attn_repre)

        # maximum attention weight over heads
        attn_weights, _ = attn_weights.max(dim=1)

        if self.return_distribute:
            return attn_repre, attn_weights, attn_weights_
        else:
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
