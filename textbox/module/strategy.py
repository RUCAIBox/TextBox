# @Time   : 2020/12/5
# @Author : Junyi LI
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/27
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def topk_sampling(logits, temperature=1.0, top_k=0, top_p=0.9):
    r"""
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits (torch.Tensor): logits distribution
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

    Return:
        torch.Tensor: the chosen index of token.
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

def greedy_search(logits):
    r"""Find the index of max logits

    Args:
        logits (torch.Tensor): logits distribution

    Return:
        torch.Tensor: the chosen index of token
    """
    return logits.argmax(dim=-1)

def beam_search(gen_idx, token_logits, completed_hypotheses, hypthetic_token_idx, hyp_scores,
                decoder_states, encoder_output=None, encoder_mask=None, beam_size, eos_token_idx, device):
    
    token_probs = F.log_softmax(token_logits, dim=-1).squeeze(1)
    vocab_size = token_probs.shape[-1]

    live_hyp_num = beam_size - len(completed_hypotheses)
    tmp_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(token_probs) + token_probs).view(-1)
    top_scores, top_pos = torch.topk(tmp_hyp_scores, k=live_hyp_num)
    hyp_ids = top_pos / vocab_size
    word_ids = top_pos % vocab_size

    new_hypotheses = []
    new_ids = []
    new_scores = []

    for hyp_id, word_id, score in zip(hyp_ids, word_ids, top_scores):
        new_hyp = hypthetic_token_idx[hyp_id] + [word_id]
        if (word_id == eos_token_idx):
            completed_hypotheses.append((new_hyp[1:-1], score / (gen_idx - 1)))
        else:
            new_hypotheses.append(new_hyp)
            new_ids.append(hyp_id)
            new_scores.append(score)

    if (len(completed_hypotheses) == beam_size):
        return completed_hypotheses, None, None, None, None, None, None

    new_ids = torch.tensor(new_ids).to(device)
    decoder_states = decoder_states[:, new_ids, :]
    hypthetic_token_idx = new_hypotheses
    hyp_scores = torch.tensor(new_scores).to(device)

    hyp_num = len(hypthetic_token_idx)
    encoder_output = encoder_output.repeat(hyp_num, 1, 1)
    encoder_mask = encoder_mask.repeat(hyp_num, 1)
    input_seq = [hyp[-1] for hyp in hypthetic_token_idx]
    input_seq = torch.tensor(input_seq).unsqueeze(1).to(device)

    return completed_hypotheses, hypthetic_token_idx, hyp_scores, input_seq, decoder_states, encoder_output, encoder_mask
