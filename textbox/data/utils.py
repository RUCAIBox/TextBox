import os
from numpy import pad
import torch
import warnings
from torch import Tensor
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from textbox import CLM_MODELS


class AbstractDataset(Dataset):
    def __init__(self, config, set):
        super().__init__()
        self.config = config
        self.set = set
        source_filename = os.path.join(config['data_path'], f'{set}.src')
        target_filename = os.path.join(config['data_path'], f'{set}.tgt')
        self.source_text = load_data(source_filename)
        self.target_text = load_data(target_filename)
        self.source_length = self.config['src_len']
        self.target_length = self.config['tgt_len']
        if set != 'test':
            assert len(self.source_text) == len(self.target_text)
    
    def __len__(self):
        return len(self.source_text)
    
    def __getitem__(self, idx):
        sample = {
            "source_text": self.source_text[idx], 
            "source_ids": self.source_ids[idx],
            "target_text": self.target_text[idx], 
        }
        if self.set != 'test':
            sample.update({
                "target_ids": self.target_ids[idx],
            })
        return sample
    
    def _init_process(self):
        if self.source_length > self.tokenizer.model_max_length:
            warnings.warn(f"The max length of source text {self.source_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and will be set to {self.tokenizer.model_max_length}.")
            self.source_length = self.tokenizer.model_max_length
            
        if self.target_length > self.tokenizer.model_max_length:
            warnings.warn(f"The max length of target text {self.target_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and will be set to {self.tokenizer.model_max_length}.")
            self.target_length = self.tokenizer.model_max_length
        
        if self.config['efficient_methods'] and 'prompt-tuning' in self.config['efficient_methods']:
            prompt_length = self.config['efficient_kwargs']['prompt_length']
            if self.config['model_name'] in CLM_MODELS:
                if self.source_length + self.target_length + prompt_length > self.tokenizer.model_max_length:
                    warnings.warn(f"The length of source text {self.source_length}, target text {self.target_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length - self.target_length}.")
                    self.source_length = self.tokenizer.model_max_length - prompt_length - self.target_length
            elif self.source_length + prompt_length > self.tokenizer.model_max_length:
                warnings.warn(f"The length of source text {self.source_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length}.")
                self.source_length = self.tokenizer.model_max_length - prompt_length

    def _process_prompt(self):
        prefix = self.config['prefix_prompt'] or ''
        suffix = self.config['suffix_prompt'] or ''

        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        self.source_max_length = self.source_length - self.tokenizer.num_special_tokens_to_add() - len(self.prefix_ids) - len(self.suffix_ids)
        self.target_max_length = self.target_length - self.tokenizer.num_special_tokens_to_add()

        if self.config['model_name'] in ['bert2bert', 'opt']:
            self.target_max_length += 1

    def tokenize(self, tokenizer):
        self.tokenizer = tokenizer
        self._init_process()
        self._process_prompt()
        self.source_ids = []
        source_ids = tokenizer(self.source_text, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)['input_ids']
        for ids in source_ids:
            ids = ids[:self.source_max_length] if self.config['truncate'] == 'tail' else ids[-self.source_max_length:]
            ids = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + ids + self.suffix_ids)
            self.source_ids.append(torch.tensor(ids, dtype=torch.long))
        if self.set != 'test':
            self.target_ids = []
            with tokenizer.as_target_tokenizer():
                target_ids = tokenizer(self.target_text, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)['input_ids']
                for ids in target_ids:
                    ids = ids[:self.target_max_length] if self.config['truncate'] == 'tail' else ids[-self.target_max_length:]
                    ids = self.tokenizer.build_inputs_with_special_tokens(ids)
                    if self.config['model_name'] in ['bert2bert', 'opt']:
                        ids = ids[1:]
                    self.target_ids.append(torch.tensor(ids, dtype=torch.long))


class AbstractCollate:
    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config['model_name'] in CLM_MODELS)
    
    def __call__(self, samples):
        batch = {}
        source_text = []
        source_ids = []
        source_mask = []
        source_length = []
        target_text = []
        source_padding_side = 'left' if self.set == 'test' and self.is_casual_model else self.tokenizer.padding_side
        
        for sample in samples:
            source_text.append(sample['source_text'])
            src_id = torch.cat([sample['source_ids'], sample['target_ids']]) if self.set != 'test' and self.is_casual_model else sample['source_ids']
            source_ids.append(src_id)
            source_mask.append(torch.ones(len(src_id), dtype=torch.long))
            source_length.append(len(src_id))
            target_text.append(sample['target_text'])
        
        batch['source_text'] = source_text
        batch['source_ids'] = _pad_sequence(source_ids, self.tokenizer.pad_token_id, source_padding_side)
        batch['source_mask'] = _pad_sequence(source_mask, 0, source_padding_side)
        batch['source_length'] = torch.tensor(source_length, dtype=torch.long)
        batch['target_text'] = target_text
        
        if self.set != 'test':
            target_ids = []
            for sample in samples:
                tgt_id = torch.cat([torch.full([len(sample['source_ids'])], -100, dtype=torch.long), sample['target_ids']]) if self.is_casual_model else sample['target_ids']
                target_ids.append(tgt_id)
            batch['target_ids'] = _pad_sequence(target_ids, -100, self.tokenizer.padding_side)
        return batch


def data_preparation(config, tokenizer):
    train_dataset = AbstractDataset(config, 'train')
    valid_dataset = AbstractDataset(config, 'valid')
    test_dataset = AbstractDataset(config, 'test')

    train_dataset.tokenize(tokenizer)
    valid_dataset.tokenize(tokenizer)
    test_dataset.tokenize(tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, collate_fn=AbstractCollate(config, tokenizer, 'train'))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['eval_batch_size'], shuffle=False, collate_fn=AbstractCollate(config, tokenizer, 'valid'))
    test_dataloader = DataLoader(test_dataset, batch_size=config['eval_batch_size'], shuffle=False, collate_fn=AbstractCollate(config, tokenizer, 'test'))
    return train_dataloader, valid_dataloader, test_dataloader

def load_data(dataset_path):
    """Load dataset from split (train, valid, test).
    This is designed for single sentence format.

    Args:
        dataset_path (str): path of dataset dir.
    
    Returns:
        List[List[str]]: the text list loaded from dataset path.
    """
    if not os.path.isfile(dataset_path):
        raise ValueError('File {} not exist'.format(os.path.abspath(dataset_path)))

    text = []
    with open(dataset_path, "r") as fin:
        for line in fin:
            l = line.strip()
            if len(l) >= 2 and ((l[0] == '"' and l[-1] == '"') or (l[0] == "'" and l[-1] == "'") or (l[0] == '[' and l[-1] == ']')):
                try:
                    l = eval(l)
                    if not isinstance(l, list):
                        l = str(l)
                except:
                    pass
            text.append(l)
    return text

def _pad_sequence(tensors: List[Tensor], padding_value: int, padding_side: str = 'right'):
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