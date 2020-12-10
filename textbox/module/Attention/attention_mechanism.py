# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class LuongAttention(torch.nn.Module):
    def __init__(self, source_size, target_size, alignment_method='concat'):
        super(LuongAttention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size
        self.alignment_method = alignment_method

        if self.alignment_method == 'general':
            self.energy_linear = nn.Linear(target_size, source_size)
        elif self.alignment_method == 'concat':
            self.energy_linear = nn.Linear(source_size + target_size, target_size)
            self.v = nn.Parameter(torch.FloatTensor(target_size))
        elif self.alignment_method == 'dot':
            assert self.source_size == target_size
        else:
            raise ValueError("The alignment method for Luong Attention must be in ['general', 'concat', 'dot'].")

    def score(self, hidden_states, encoder_outputs):
        tgt_len = hidden_states.size(1)
        src_len = encoder_outputs.size(1)

        if self.alignment_method == 'general':
            energy = self.energy_linear(hidden_states)  # B * tgt_len * src_size
            encoder_outputs = encoder_outputs.permute(0, 2, 1)  # B * src_size * src_len
            energy = energy.bmm(encoder_outputs)  # B * tgt_len * src_len
            return energy
        elif self.alignment_method == 'concat':
            hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, src_len, 1)  # B * tgt_len * src_len * target_size
            encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, tgt_len, 1, 1)
            energy = torch.tanh(self.energy_linear(torch.cat((hidden_states, encoder_outputs), dim=-1)))
            energy = self.v.mul(energy).sum(dim=-1)
            return energy
        elif self.alignment_method == 'dot':
            encoder_outputs = encoder_outputs.permute(0, 2, 1)
            energy = hidden_states.bmm(encoder_outputs)
            return energy
        else:
            raise NotImplementedError(
                "No such alignment method {} for computing Luong scores.".format(self.alignment_method))

    def forward(self, hidden_states, encoder_outputs, encoder_masks):
        """
        :param hidden_states: B * tgt_len * hidden_size
        :param encoder_outputs: B * src_len * hidden_size
        :param encoder_masks: B * src_len
        :return:
        """
        tgt_len = hidden_states.size(1)
        energy = self.score(hidden_states, encoder_outputs)
        probs = F.softmax(energy, dim=-1) * encoder_masks.unsqueeze(1).repeat(1, tgt_len, 1)
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        context = probs.bmm(encoder_outputs)
        return context, probs


class BahdanauAttention(torch.nn.Module):
    def __init__(self, source_size, target_size):
        super(BahdanauAttention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size

        self.energy_linear = nn.Linear(source_size + target_size, target_size)
        self.v = nn.Parameter(torch.FloatTensor(target_size))

    def score(self, hidden_states, encoder_outputs):
        """
        :param hidden_states: B * target_size
        :param encoder_outputs: B * src_len * source_size
        :return:
        """
        src_len = encoder_outputs.size(1)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, src_len, 1)  # B * src_len * target_size
        # print(hidden_states.size(), encoder_outputs.size())

        energy = torch.tanh(self.energy_linear(torch.cat((hidden_states, encoder_outputs), dim=-1)))
        energy = self.v.mul(energy).sum(dim=-1)
        return energy

    def forward(self, hidden_states, encoder_outputs, encoder_masks):
        """
        :param hidden_states: B * target_size
        :param encoder_outputs: B * src_len * source_size
        :param encoder_masks: B * src_len
        :return:
            context: B * 1 * source_size
        """
        energy = self.score(hidden_states, encoder_outputs)
        probs = F.softmax(energy, dim=-1) * encoder_masks
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        probs = probs.unsqueeze(1)
        context = probs.bmm(encoder_outputs)

        return context, probs


class MonotonicAttention(torch.nn.Module):
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
        """Additive gaussian nosie to encourage discreteness"""
        return torch.FloatTensor(*size).normal_()

    def safe_cumprod(self, x):
        """Numerically stable cumulative product by cumulative sum in log-space"""
        return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1))

    def exclusive_cumprod(self, x):
        """Exclusive cumulative product [a, b, c] => [1, a, a * b]"""
        batch_size = x.size(0)
        ones = torch.ones(batch_size, 1).to(x.device)
        one_x = torch.cat((ones, x), dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def score(self, hidden_states, encoder_outputs):
        tgt_len = hidden_states.size(1)
        src_len = encoder_outputs.size(1)
        energy = torch.tanh(self.w_linear(encoder_outputs).unsqueeze(1).repeat(1, tgt_len, 1, 1) +
                            self.v_linear(hidden_states).unsqueeze(2).repeat(1, 1, src_len, 1) +
                            self.bias)
        energy = self.v(energy).squeeze(-1) + self.r
        return energy

    def soft(self, hidden_states, encoder_outputs, encoder_masks, previous_probs=None):
        r"""
        Soft monotonic attention (Train)

        Args:
            hidden_states: [batch_size, tgt_len, target_size]
            encoder_outputs [batch_size, src_len, source_size]
            encoder_masks [batch_size, src_len]
            previous_alpha [batch_size, tgt_len, src_len]

        Return:
            probs [batch_size, tgt_len, src_len]
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
        """
        Hard monotonic attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
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
    def __init__(self, embedding_size, num_heads, attn_weight_dropout_ratio=0.0):
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
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
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
