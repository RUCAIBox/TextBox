# @Time   : 2020/12/5
# @Author : Junyi LI
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/27
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
textbox.module.strategy
#############################
Common Strategys in text generation
"""

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
    return logits.squeeze(1).argmax(dim=-1)


class Beam_Search_Hypothesis(object):
    r""" Class designed for beam search.
    """

    def __init__(self, beam_size, sos_token_idx, eos_token_idx, device, idx2token):
        self.beam_size = beam_size
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.device = device
        self.idx2token = idx2token

        self.hypthetic_token_idx = [[sos_token_idx]]
        self.completed_hypotheses = []
        self.hyp_scores = torch.zeros(1).to(device)

    def generate(self):
        r""" Pick the hypothesis with max prob among beam_size hypothesises.

        Return:
            List[str]: the generated tokens
        """
        generate_idx = self.hypthetic_token_idx[0][1:] if (len(
            self.completed_hypotheses
        ) == 0) else max(self.completed_hypotheses, key=lambda hyp: hyp[1])[0]
        generate_tokens = [self.idx2token[idx.item()] for idx in generate_idx]
        return generate_tokens

    def stop(self):
        r""" Determine if the beam search is over.

        Return:
            Bool: ``True`` represents the search over, `Flase` represents the search working.
        """
        return len(self.completed_hypotheses) == self.beam_size

    def step(
        self, gen_idx, token_logits, decoder_states=None, encoder_output=None, encoder_mask=None, input_type='token'
    ):
        r""" A step for beam search.

        Args:
            gen_idx (int): the generated step number.
            token_logits (torch.Tensor): logits distribution, shape: [hyp_num, sequence_length, vocab_size].
            decoder_states (torch.Tensor, optional): the states of decoder needed to choose, shape: [hyp_num, sequence_length, hidden_size], default: None.
            encoder_output (torch.Tensor, optional): the output of encoder needed to copy, shape: [hyp_num, sequence_length, hidden_size], default: None.
            encoder_mask (torch.Tensor, optional): the mask of encoder to copy, shape: [hyp_num, sequence_length], default: None.

        Return:
            torch.Tensor: the next input squence, shape: [hyp_num],
            torch.Tensor, optional: the chosen states of decoder, shape: [new_hyp_num, sequence_length, hidden_size]
            torch.Tensor, optional: the copyed output of encoder, shape: [new_hyp_num, sequence_length, hidden_size]
            torch.Tensor, optional: the copyed mask of encoder, shape: [new_hyp_num, sequence_length]
        """
        token_probs = F.log_softmax(token_logits, dim=-1).squeeze(1)
        vocab_size = token_probs.shape[-1]

        live_hyp_num = self.beam_size - len(self.completed_hypotheses)
        tmp_hyp_scores = (self.hyp_scores.unsqueeze(1).expand_as(token_probs) + token_probs).view(-1)
        top_scores, top_pos = torch.topk(tmp_hyp_scores, k=live_hyp_num)
        hyp_ids = (top_pos / vocab_size).long()
        word_ids = top_pos % vocab_size

        new_hypotheses = []
        new_ids = []
        new_scores = []

        for hyp_id, word_id, score in zip(hyp_ids, word_ids, top_scores):
            new_hyp = self.hypthetic_token_idx[hyp_id] + [word_id]
            if (word_id == self.eos_token_idx):
                self.completed_hypotheses.append((new_hyp[1:-1], score / (gen_idx - 1)))
            else:
                new_hypotheses.append(new_hyp)
                new_ids.append(hyp_id)
                new_scores.append(score)

        if (len(self.completed_hypotheses) == self.beam_size):
            none_cnt = (decoder_states is not None) + (encoder_output is not None) + (encoder_mask is not None) + 1
            return [None] * none_cnt

        self.hypthetic_token_idx = new_hypotheses
        self.hyp_scores = torch.tensor(new_scores).to(self.device)

        hyp_num = len(self.hypthetic_token_idx)
        if (input_type == 'token'):
            input_seq = [hyp[-1] for hyp in self.hypthetic_token_idx]
            input_seq = torch.tensor(input_seq).unsqueeze(1).to(self.device)
        elif (input_type == 'whole'):
            input_seq = torch.tensor(self.hypthetic_token_idx).to(self.device)
        else:
            raise ValueError("The input type must be in ['token', 'whole'].")

        returns = [input_seq]

        if (decoder_states is not None):
            new_ids = torch.tensor(new_ids).to(self.device)
            if (isinstance(decoder_states, tuple)):
                (x, y) = decoder_states
                decoder_states = (x[:, new_ids, :], y[:, new_ids, :])
            else:
                decoder_states = decoder_states[:, new_ids, :]
            returns += [decoder_states]

        if (encoder_output is not None):
            encoder_output = encoder_output[0:1].repeat(hyp_num, 1, 1)
            returns += [encoder_output]

        if (encoder_mask is not None):
            encoder_mask = encoder_mask[0:1].repeat(hyp_num, 1)
            returns += [encoder_mask]

        return returns
    
def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
    r"""
    Given scores from an expanded dictionary
    corresponding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)
    for b in range(batch.batch_size):
        blank = []
        fill = []
        index = batch.indices.data[b]
        src_vocab = src_vocabs[index]
        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            scores[:, b].index_add_(1, fill,
                                    scores[:, b].index_select(1, blank))
            scores[:, b].index_fill_(1, blank, 1e-10)
    return scores

class CopyGenerator(nn.Module):
    r"""Generator module that additionally considers copying words directly from the source.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary

    """
    def __init__(self, input_size, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, len(tgt_dict)) # hidden_size -> tgt_vocab_size
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, src_map=None, align=None, ptrs=None):
        r"""
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()

        # Original probabilities.
        logits = self.linear(hidden) 
        logits[:, 0] = -float('inf')
        prob = F.softmax(logits,dim=1) 
        p_copy = torch.sigmoid(self.linear_copy(hidden))

        if self.training:
            align_unk = align.eq(0).float().view(-1, 1)
            align_not_unk = align.ne(0).float().view(-1, 1)
            out_prob = torch.mul(prob, align_unk.expand_as(prob)) 
            mul_attn = torch.mul(attn, align_not_unk.expand_as(attn))
        else:
            out_prob = torch.mul(prob,  1 - p_copy.expand_as(prob))
            mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1),
                              src_map.transpose(0, 1)).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)

        return torch.cat([out_prob, copy_prob], 1), p_copy

class CopyGeneratorCriterion(object):
    """
    Calculate loss in copy mechanism
    """
    def __init__(self, vocab_size,  pad, eps=1e-20):

        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target):
        # Compute unks in align and target for readability
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)
        out = out + tmp.mul(align_unk)
        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float())
        return loss
