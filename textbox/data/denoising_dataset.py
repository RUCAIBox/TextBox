import os, math
import torch
import numpy as np
from numpy.random import permutation, poisson
from logging import getLogger
from textbox import SpecialTokens
from typing import Optional, List, Union, Dict, Tuple
from textbox import CLM_MODELS
from torch.nn.utils.rnn import pad_sequence


def _pad_sequence(
    tensors: List[torch.Tensor], padding_value: int, padding_side: str = "right"
):
    """
    Pad encoded inputs (on left/right and up to max length in the batch)
    """
    max_len = max(tensor.size(0) for tensor in tensors)
    padded_tensors = []
    if padding_side == "right":
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    elif padding_side == "left":
        for tensor in tensors:
            padding_length = max_len - len(tensor)
            padded_tensor = torch.cat(
                [
                    torch.full([padding_length], padding_value, dtype=tensor.dtype),
                    tensor,
                ],
                dim=-1,
            )
            padded_tensors.append(padded_tensor)
    padded_tensors = torch.stack(padded_tensors, dim=0)
    return padded_tensors


def _collate_batch(samples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `samples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(samples[0], (list, tuple)):
        samples = [torch.tensor(e, dtype=torch.long) for e in samples]

    # Check if padding is necessary.
    length_of_first = samples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in samples)
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        return torch.stack(samples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in samples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = samples[0].new_full([len(samples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(samples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class DenoisingCollate:
    """Data collator used denoising language modeling task in BART.
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    
    BART Original hyperparams: https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320
    """

    mask_ratio: float = 0.3
    poisson_lambda: float = 3.5
    permutate_sentence_ratio: float = 1.0
    pad_to_multiple_of: int = 16

    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            samples (dict): list of samples each samples contains input_ids field
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {}
        source_text = []
        source_ids = []
        source_mask = []
        source_length = []
        target_text = []
        source_padding_side = (
            "left"
            if self.set == "test" and self.is_casual_model
            else self.tokenizer.padding_side
        )

        for sample in samples:
            source_text.append(sample["source_text"])
            src_id = (
                torch.cat([sample["source_ids"], sample["target_ids"]])
                if self.set != "test" and self.is_casual_model
                else sample["source_ids"]
            )
            source_ids.append(src_id)
            source_mask.append(torch.ones(len(src_id), dtype=torch.long))
            source_length.append(len(src_id))
            target_text.append(sample["target_text"])

        batch["source_text"] = source_text
        batch["source_ids"] = _pad_sequence(
            source_ids, self.tokenizer.pad_token_id, source_padding_side
        )
        batch["source_mask"] = _pad_sequence(source_mask, 0, source_padding_side)
        batch["source_length"] = torch.tensor(source_length, dtype=torch.long)
        batch["target_text"] = target_text
        # If special token mask has been preprocessed,
        # batch["decoder_input_ids"] = self.shift_tokens_right(batch["input_ids"])

        do_permutate = False
        batch["source_ids"] = batch["source_ids"].cpu().detach().numpy()
        if self.permutate_sentence_ratio > 0.0:
            batch["source_ids"] = self.permutate_sentences(batch["source_ids"])
            do_permutate = True

        if self.mask_ratio:
            batch["source_ids"], batch["target_ids"] = self.add_whole_word_mask(
                batch["source_ids"], do_permutate
            )

        if isinstance(batch["source_ids"], np.ndarray):
            batch["source_ids"] = torch.tensor(batch["source_ids"])

        if isinstance(batch["target_ids"], np.ndarray):
            batch["target_ids"] = torch.tensor(batch["target_ids"])

        return batch

    def permutate_sentences(self, inputs):
        results = inputs.copy()

        full_stops = inputs == self.tokenizer.eos_token_id

        sentence_ends = np.argwhere(full_stops[:, 1:] * ~full_stops[:, :-1])
        sentence_ends[:, 1] += 2
        num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)[1]
        num_to_permute = np.ceil(
            (num_sentences * 2 * self.permutate_sentence_ratio) / 2.0
        ).astype(int)

        sentence_ends = np.split(
            sentence_ends[:, 1],
            np.unique(sentence_ends[:, 0], return_index=True)[1][1:],
        )

        for i in range(inputs.shape[0]):
            substitutions = np.random.permutation(num_sentences[i])[: num_to_permute[i]]

            ordering = np.arange(0, num_sentences[i])
            ordering[substitutions] = substitutions[
                np.random.permutation(num_to_permute[i])
            ]

            index = 0
            for j in ordering:
                sentence = inputs[
                    i, (sentence_ends[i][j - 1] if j > 0 else 0) : sentence_ends[i][j]
                ]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return torch.tensor(inputs), torch.tensor(labels)

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate(
                [lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))]
            )

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        token_indices = np.argwhere(is_token == 1)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id

        if not do_permutate:
            labels[np.where(mask)] = -100
        else:
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(
            np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)
        ):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels


class TextInfillingCollate:
    # Original BART pretraining hyperparams: https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320
    mlm_probability: float = 0.3
    poisson_lambda: float = 3.5
    pad_to_multiple_of: Optional[int] = None

    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError

    def __call__(
        self, samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # padded_encoder_input = _collate_batch(samples['source_ids'], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        batch = {}
        source_text = []
        source_ids = []
        source_mask = []
        source_length = []
        target_text = []
        source_padding_side = (
            "left"
            if self.set == "test" and self.is_casual_model
            else self.tokenizer.padding_side
        )

        for sample in samples:
            source_text.append(sample["source_text"])
            src_id = (
                torch.cat([sample["source_ids"], sample["target_ids"]])
                if self.set != "test" and self.is_casual_model
                else sample["source_ids"]
            )
            source_ids.append(src_id)
            source_mask.append(torch.ones(len(src_id), dtype=torch.long))
            source_length.append(len(src_id))
            target_text.append(sample["target_text"])

        batch["source_text"] = source_text
        batch["source_ids"] = _pad_sequence(
            source_ids, self.tokenizer.pad_token_id, source_padding_side
        )
        batch["source_mask"] = _pad_sequence(source_mask, 0, source_padding_side)
        batch["source_length"] = torch.tensor(source_length, dtype=torch.long)
        batch["target_text"] = target_text
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["source_ids"], batch["target_ids"] = self.mask_tokens(
            batch["source_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # determine how many tokens we need to mask in total
        is_token = ~(inputs == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum() * self.mlm_probability))

        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)
        lengths = poisson_distribution.sample(sample_shape=(num_to_mask,))
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat(
                [lengths, poisson_distribution.sample(sample_shape=(num_to_mask,))]
            )

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        token_indices = is_token.nonzero(as_tuple=False)
        span_starts = torch.randperm(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(inputs, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = inputs.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[special_tokens_mask] = False
        inputs[mask.bool()] = self.tokenizer.mask_token_id
        labels[~mask.bool()] = -100

        # remove mask tokens that are not starts of spans
        to_remove = mask.bool() & mask.bool().roll(1, 1)
        new_inputs = torch.full_like(inputs, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(
            torch.split(inputs, split_size_or_sections=1, dim=0)
        ):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        return new_inputs, labels
