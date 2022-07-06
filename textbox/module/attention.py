import torch
from torch import nn
import torch.nn.functional as F


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
