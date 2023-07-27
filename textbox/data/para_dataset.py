import os
import torch, warnings
import math
from torch.utils.data import DataLoader, Dataset
from textbox import CLM_MODELS
from typing import List
from torch.nn.utils.rnn import pad_sequence
from textbox.data.misc import load_data, _pad_sequence
import thulac
import re
import random

MAX_TOKENIZE_NUM = 1000000
def find_align_pairs(thu, src, tgt):
    punc_list = ["，", "。", "：", "；", "“", "”", "！", "？", "《", "》", "’", "‘", "\n"]
    tgt_words=thu.cut(tgt, text=True)
    tgt_words=[i for i in tgt_words.split(" ") if i not in punc_list]
    align_pairs = []

    for idx, src_token in enumerate(list(src)):
        if src_token in punc_list:
            continue
        for tgt_word in tgt_words:
            if len(tgt_word) != 2:
                continue
            if src_token in tgt_word:
                if (tgt_word not in src):
                    new_pair = (idx, tgt_word)
                    align_pairs.append(new_pair)
    
    return align_pairs

def my_mask(tokenizer, inputs, mask_ratio):
    bsz, seq_len = inputs.size()
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    num_to_mask = math.ceil((~special_tokens_mask).sum() * mask_ratio)

    token_indices = (~special_tokens_mask).nonzero()
    rand_index = torch.randperm(token_indices.size(0))
    rand_index  = rand_index[:num_to_mask] #生成了要mask的位置

    #生成labels，和src_ids相同尺寸的-100，torch.full_like，把其中rand_index位置变成src_id
    labels=torch.full_like(inputs, -100)
    labels[tuple(token_indices[rand_index].t())]=inputs[tuple(token_indices[rand_index].t())]

    #811:
    mask_index  = rand_index[:int(0.8*num_to_mask)]
    replace_index=rand_index[int(0.8*num_to_mask):int(0.9*num_to_mask)]

    #让inputs对应位置变成mask_id
    if len(mask_index)!=0:
        inputs[tuple(token_indices[mask_index].t())]=tokenizer.mask_token_id

    #replace：torch.multinomial，生成replace_index长度的，每个词为词表随机生成的内容
    if len(replace_index)!=0:
        vocab=torch.tensor(list(dict(tokenizer.vocab).values()), dtype=float)
        inputs[tuple(token_indices[replace_index].t())]=torch.multinomial(vocab, len(replace_index), replacement=True)

    inputs = _pad_sequence(inputs, tokenizer.pad_token_id)
    return inputs, labels

class TranslateDataset(Dataset):
    def __init__(self, config, set):
        super().__init__()
        self.config = config
        self.quick_test = config["quick_test"]
        if isinstance(self.quick_test, bool):
            self.quick_test = 32 if self.quick_test else 0
        self.set = set
        source_filename = os.path.join(config["data_path"], f"{set}.src")
        target_filename = os.path.join(config["data_path"], f"{set}.tgt")

        self.source_text = load_data(source_filename, max_length=self.quick_test)
        self.pretraining = config['pretrain_task']
        self.target_text = load_data(target_filename, max_length=self.quick_test)
        self.source_length = self.config["src_len"]
        self.target_length = self.config["tgt_len"]
        self.paired_text = bool(
            set == "train" or (self.set == 'valid' and self.config['metrics_for_best_model'] == ['loss'])
        )
        # if self.paired_text:
        #     assert len(self.source_text) == len(self.target_text)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, idx):
        sample = {
            "source_text": self.source_text[idx],
        }
        sample.update({"source_ids": self.source_ids[idx], "target_text": self.target_text[idx]})
        if self.paired_text:
            sample.update({"target_ids": self.target_ids[idx]})
        return sample

    def _init_process(self):
        if self.source_length > self.tokenizer.model_max_length:
            warnings.warn(
                f"The max length of source text {self.source_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and will be set to {self.tokenizer.model_max_length}."
            )
            self.source_length = self.tokenizer.model_max_length

        if self.target_length > self.tokenizer.model_max_length:
            warnings.warn(
                f"The max length of target text {self.target_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and will be set to {self.tokenizer.model_max_length}."
            )
            self.target_length = self.tokenizer.model_max_length
        if self.config["model_name"] in ["unilm"] + CLM_MODELS:
            if self.source_length + self.target_length > self.tokenizer.model_max_length:
                tgt_len = math.floor(self.target_length / (self.source_length + self.target_length) * self.tokenizer.model_max_length)
                src_len = self.tokenizer.model_max_length - tgt_len
                warnings.warn(
                    f"The max length of the sum of source text {self.source_length} and target text {self.target_length}"
                    f" exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, "
                    f"and will be set to {src_len} and {tgt_len}, respectively."
                )
                self.target_length = tgt_len
                self.source_length = src_len

        if (self.config["efficient_methods"] and "prompt-tuning" in self.config["efficient_methods"]):
            prompt_length = self.config["efficient_kwargs"]["prompt_length"]
            if self.config["model_name"] in CLM_MODELS:
                if (self.source_length + self.target_length + prompt_length > self.tokenizer.model_max_length):
                    warnings.warn(
                        f"The length of source text {self.source_length}, target text {self.target_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length - self.target_length}."
                    )
                    self.source_length = (self.tokenizer.model_max_length - prompt_length - self.target_length)
            elif self.source_length + prompt_length > self.tokenizer.model_max_length:
                warnings.warn(
                    f"The length of source text {self.source_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length}."
                )
                self.source_length = self.tokenizer.model_max_length - prompt_length

    def _process_prompt(self):
        prefix = self.config["prefix_prompt"] or ""
        suffix = self.config["suffix_prompt"] or ""

        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        self.source_max_length = self.source_length - self.tokenizer.num_special_tokens_to_add() \
                                 - len(self.prefix_ids) - len(self.suffix_ids)
        self.target_max_length = self.target_length - self.tokenizer.num_special_tokens_to_add()

        if self.config["model_name"] in ["bert2bert", "opt", "unilm", "xlm"]:
            self.target_max_length += 1

    def tokenize(self, tokenizer):
        self.tokenizer = tokenizer
        self._init_process()
        self._process_prompt()

        self.source_ids = []
        source_ids = []
        bsz_num = math.ceil(len(self.source_text) / MAX_TOKENIZE_NUM)
        for i in range(bsz_num):
            ids = tokenizer(
                self.source_text[i * MAX_TOKENIZE_NUM:(i + 1) * MAX_TOKENIZE_NUM],
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )["input_ids"]
            source_ids.extend(ids)
        for ids in source_ids:
            ids = ids[:self.source_max_length] if self.config["truncate"] == "tail" else ids[-self.source_max_length:]
            ids = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + ids + self.suffix_ids)
            self.source_ids.append(torch.tensor(ids, dtype=torch.long))

        if self.paired_text:
            self.target_ids = []
            target_ids = tokenizer(
                text_target=self.target_text,
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )["input_ids"]
            for ids in target_ids:
                if self.config["truncate"] == "tail":
                    ids = ids[:self.target_max_length]
                else:
                    ids = ids[-self.target_max_length:]
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
                if self.config["model_name"] in ["bert2bert", "opt", "unilm", "xlm"]:
                    ids = ids[1:]
                self.target_ids.append(torch.tensor(ids, dtype=torch.long))

class ParaCollate:
    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)
        self.paired_text = bool(
            set == "train" or (self.set == 'valid' and self.config['metrics_for_best_model'] == ['loss'])
        )
        self.thu=thulac.thulac(seg_only=True)

        self.mask_ratio = config['mask_ratio'] or 0.15
        self.poisson_lambda = config['poisson_lambda'] or 3.5
        self.poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)

    @classmethod
    def get_type(cls) -> str:
        return 'pretrain parallel (UNI & BI dec)'

    def __call__(self, samples):
        batch = {}

        my_mask_ratio=0.7
        #my_mask_ratio=0.7
        batch1 = {}
        source_text = []
        source_ids = []
        source_mask = []
        source_length = []
        target_text = []
        source_padding_side = ("left" if not self.paired_text and self.is_casual_model else self.tokenizer.padding_side)

        for sample in samples:
            src=sample["source_text"]
            tgt=sample["target_text"]
            align_pairs=find_align_pairs(self.thu, src, tgt)
            src = list(src)
            for a in align_pairs:
                tem=random.uniform(0, 1)
                if tem<my_mask_ratio:
                    src[a[0]] =a[1]
            src=''.join(src)
            source_text.append(src)

        my_source_id = self.my_tokenize(source_text)

        for i in range(len(samples)):
            sample=samples[i]
            if self.paired_text and self.is_casual_model:
                src_id = torch.cat([my_source_id[i], sample["target_ids"]])
            else:
                src_id = my_source_id[i]
            source_ids.append(src_id)
            source_mask.append(torch.ones(len(src_id), dtype=torch.long))
            source_length.append(len(src_id))
            target_text.append(sample["target_text"])
        
        batch1["source_text"] = source_text
        batch1["source_ids"] = _pad_sequence(source_ids, self.tokenizer.pad_token_id, source_padding_side)
        batch1["source_mask"] = _pad_sequence(source_mask, 0, source_padding_side)
        batch1["source_length"] = torch.tensor(source_length, dtype=torch.long)
        batch1["target_text"] = target_text

        if self.paired_text:
            target_ids = []
            for sample in samples:
                if self.is_casual_model:
                    tgt_id = torch.cat([
                        torch.full([len(sample["source_ids"])], -100, dtype=torch.long), sample["target_ids"]
                    ])
                else:
                    tgt_id = sample["target_ids"]
                target_ids.append(tgt_id)
            batch1["target_ids"] = _pad_sequence(target_ids, -100, self.tokenizer.padding_side)

        batch2={}
        source_text = [sample["source_text"] for sample in samples]
        source_ids = self.tokenizer(
            source_text,
            max_length=self.config['src_len'],
            truncation=True,
            padding=True,
            return_attention_mask=False,
            return_tensors='pt'
        )['input_ids']

        target_text = [sample["target_text"] for sample in samples]
        target_ids = self.tokenizer(
            target_text,
            max_length=self.config['tgt_len'],
            truncation=True,
            padding=True,
            return_attention_mask=False,
            return_tensors='pt'
        )['input_ids']

        if self.mask_ratio > 0.0:
            #mask_ratio=random.uniform(0.1, 0.2)
            mask_ratio=0.15
            source_ids, labels_enc = my_mask(self.tokenizer, source_ids, mask_ratio)
            #mask_ratio=random.uniform(0.2, 0.5)
            mask_ratio=0.35
            target_ids, labels_dec = my_mask(self.tokenizer, target_ids, mask_ratio)

        batch2["source_ids"] = source_ids
        batch2["source_mask"] = source_ids.ne(self.tokenizer.pad_token_id)
        batch2["enc_labels"]=labels_enc
        
        batch2["target_ids"] = target_ids
        batch2["target_mask"] = target_ids.ne(self.tokenizer.pad_token_id)
        batch2["dec_labels"]=labels_dec

        batch['uni']=batch1
        batch['bi']=batch2
        batch['is_stage1']=False
        batch['is_stage2']=True
        return batch

    def _init_process(self):
        if self.config["src_len"] > self.tokenizer.model_max_length:
            self.config["src_len"] = self.tokenizer.model_max_length

    def _process_prompt(self):
        prefix = self.config["prefix_prompt"] or ""
        suffix = self.config["suffix_prompt"] or ""

        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        self.source_max_length = self.config["src_len"] - self.tokenizer.num_special_tokens_to_add() \
                                 - len(self.prefix_ids) - len(self.suffix_ids)

    def my_tokenize(self, my_text):
        self._init_process()
        self._process_prompt()

        s_source_ids = []
        source_ids = []
        bsz_num = math.ceil(len(my_text) / MAX_TOKENIZE_NUM)
        for i in range(bsz_num):
            ids = self.tokenizer(
                my_text[i * MAX_TOKENIZE_NUM:(i + 1) * MAX_TOKENIZE_NUM],
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )["input_ids"]
            source_ids.extend(ids)
        for ids in source_ids:
            ids = ids[:self.source_max_length] if self.config["truncate"] == "tail" else ids[-self.source_max_length:]
            ids = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + ids + self.suffix_ids)
            s_source_ids.append(torch.tensor(ids, dtype=torch.long))
        return s_source_ids
 