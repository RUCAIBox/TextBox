# @Time   : 2020/12/5
# @Author : Junyi LI
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def topk_sampling(logits, temperature=1.0, top_k=0, top_p=0.9):
    r"""
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        values = torch.topk(logits, top_k)[0]  # B x top_k
        batch_mins = values[:, :, -1].expand_as(logits.squeeze(1)).unsqueeze(1)
        logits = torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

    if 0.0 < top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

        cumprobs = sorted_probs.cumsum(dim=-1)

        # Create mask for all cumulative probabilities less than p
        mask = cumprobs < top_p

        # First mask must always be pickable
        mask = F.pad(mask[:, :, :-1], (1, 0, 0, 0), value=1)

        masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')).to(probs))

        batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)

        # Mask out all logits (tail) that are too small
        logits = torch.where(probs < batch_mins, torch.tensor(float('-inf')).to(logits), logits)

    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities.squeeze(1)
    token_idx = torch.multinomial(probabilities, 1)

    return token_idx

