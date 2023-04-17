import os
import torch, warnings
import math
from torch.utils.data import DataLoader, Dataset
from textbox import CLM_MODELS
from typing import List
from torch.nn.utils.rnn import pad_sequence
from textbox.data.misc import load_data, _pad_sequence

MAX_TOKENIZE_NUM = 1000000


class AbstractDataset(Dataset):

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
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)
        if self.pretraining is None and self.pretraining != 'disabled':
            self.target_text = load_data(target_filename, max_length=self.quick_test)
        self.source_length = self.config["src_len"]
        self.target_length = self.config["tgt_len"]
        self.paired_text = bool(
            set == "train" or (self.set == 'valid' and self.config['metrics_for_best_model'] == ['loss'])
        )
        if self.paired_text and self.pretraining is None:
            assert len(self.source_text) == len(self.target_text)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, idx):
        sample = {
            "source_text": self.source_text[idx],
        }
        if self.pretraining is None:
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
                tgt_len = math.floor(
                    self.target_length / (self.source_length + self.target_length) * self.tokenizer.model_max_length
                )
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

        self.config["src_len"] = self.source_length
        self.config["tgt_len"] = self.target_length

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

        if self.pretraining:
            return

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
            ids = self.prefix_ids + ids + self.suffix_ids
            if not self.is_casual_model:
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
            self.source_ids.append(torch.tensor(ids, dtype=torch.long))

        if self.paired_text and self.pretraining is None:
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


class AbstractCollate:

    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)
        self.paired_text = bool(
            set == "train" or (self.set == 'valid' and self.config['metrics_for_best_model'] == ['loss'])
        )

    @classmethod
    def get_type(cls) -> str:
        return 'pretrain disabled'

    def __call__(self, samples):
        batch = {}
        source_text = []
        source_ids = []
        source_mask = []
        source_length = []
        target_text = []
        source_padding_side = ("left" if not self.paired_text and self.is_casual_model else self.tokenizer.padding_side)

        for sample in samples:
            source_text.append(sample["source_text"])
            if self.paired_text and self.is_casual_model:
                src_id = torch.cat([sample["source_ids"], sample["target_ids"]])
            else:
                src_id = sample["source_ids"]
            source_ids.append(src_id)
            source_mask.append(torch.ones(len(src_id), dtype=torch.long))
            source_length.append(len(src_id))
            target_text.append(sample["target_text"])

        batch["source_text"] = source_text
        batch["source_ids"] = _pad_sequence(source_ids, self.tokenizer.pad_token_id, source_padding_side)
        batch["source_mask"] = _pad_sequence(source_mask, 0, source_padding_side)
        batch["source_length"] = torch.tensor(source_length, dtype=torch.long)
        batch["target_text"] = target_text

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
            batch["target_ids"] = _pad_sequence(target_ids, -100, self.tokenizer.padding_side)
        return batch
