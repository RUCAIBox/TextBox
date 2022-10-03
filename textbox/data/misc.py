from typing import List, Optional
import itertools, os
import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(dataset_path: str, max_length: int = 0):
    """Load dataset from split (train, valid, test).
    This is designed for single sentence format.

    Args:
        dataset_path (str): path of dataset dir.
        max_length: (default = 0) The amount of line about to load. If 0, quick_test will be disabled.

    Returns:
        List[List[str]]: the text list loaded from dataset path.
    """
    if not os.path.isfile(dataset_path):
        raise ValueError('File {} not exist'.format(os.path.abspath(dataset_path)))

    text = []
    with open(dataset_path, "r") as fin:
        if max_length:
            fin = itertools.islice(fin, max_length)
        for line in fin:
            l = line.strip()
            if len(l) >= 2 and ((l[0] == '"' and l[-1] == '"') or (l[0] == "'" and l[-1] == "'") or
                                (l[0] == '[' and l[-1] == ']')):
                try:
                    l = eval(l)
                    if not isinstance(l, list):
                        l = str(l)
                except:
                    pass
            text.append(l)
    return text


def _pad_sequence(tensors: List[torch.Tensor], padding_value: int, padding_side: str = 'right'):
    """
    Pad encoded inputs (on left/right and up to max length in the batch)
    """
    max_len = max(tensor.size(0) for tensor in tensors)
    padded_tensors = []
    if padding_side == 'right':
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    elif padding_side == 'left':
        for tensor in tensors:
            padding_length = max_len - len(tensor)
            padded_tensor = torch.cat([torch.full([padding_length], padding_value, dtype=tensor.dtype), tensor], dim=-1)
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
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
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
            result[i, :example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result
