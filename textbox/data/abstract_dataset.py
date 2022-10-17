import os
import torch, warnings
import random
from torch.utils.data import DataLoader, Dataset
from textbox import CLM_MODELS
from typing import List
from torch.nn.utils.rnn import pad_sequence
from textbox.data.misc import load_data, _pad_sequence


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
        self.target_text = load_data(target_filename, max_length=self.quick_test)
        self.source_length = self.config["src_len"]
        self.target_length = self.config["tgt_len"]
        if set == "train":
            assert len(self.source_text) == len(self.target_text)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, idx):
        sample = {
            "source_text": self.source_text[idx],
            "source_ids": self.source_ids[idx],
            "target_text": self.target_text[idx],
        }
        if self.set == "train":
            sample.update({
                "target_ids": self.target_ids[idx],
            })
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
        if self.config["model_name"] == "unilm":
            if self.source_length + self.target_length > self.tokenizer.model_max_length:
                warnings.warn(
                    f"The max length of the sum of source text {self.source_length} and target text {self.target_length}"
                    f" exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, "
                    f"and will be set to {self.tokenizer.model_max_length // 2} and "
                    f"{self.tokenizer.model_max_length - self.tokenizer.model_max_length // 2}."
                )
                self.target_length = self.tokenizer.model_max_length // 2
                self.source_length = self.tokenizer.model_max_length - self.target_length

        if (self.config["efficient_methods"] and "prompt-tuning" in self.config["efficient_methods"]):
            prompt_length = self.config["efficient_kwargs"]["prompt_length"]
            if self.config["model_name"] in CLM_MODELS:
                if (self.source_length + self.target_length + prompt_length > self.tokenizer.model_max_length):
                    warnings.warn(
                        f"The length of source text {self.source_length}, target text {self.target_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length - self.target_length}."
                    )
                    self.source_length = (self.tokenizer.model_max_length - prompt_length - self.target_length)
                    self.source_length = (
                            self.tokenizer.model_max_length
                            - prompt_length
                            - self.target_length
                    )
            elif self.source_length + prompt_length > self.tokenizer.model_max_length:
                warnings.warn(
                    f"The length of source text {self.source_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length}."
                )
                self.source_length = self.tokenizer.model_max_length - prompt_length

    def _process_prompt(self):
        prefix = self.config["prefix_prompt"] or ""
        suffix = self.config["suffix_prompt"] or ""

        if self.config['model_name'] == 'unilm':
            prefix = ""

        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        self.source_max_length = (
            self.source_length - self.tokenizer.num_special_tokens_to_add() - len(self.prefix_ids) -
            len(self.suffix_ids)
        )
        self.target_max_length = (self.target_length - self.tokenizer.num_special_tokens_to_add())

        if self.config["model_name"] in ["bert2bert", "opt", "unilm"]:
            self.target_max_length += 1

    def tokenize(self, tokenizer):
        self.tokenizer = tokenizer
        self._init_process()
        self._process_prompt()
        self.source_ids = []
        source_ids = tokenizer(
            self.source_text,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"]
        for ids in source_ids:
            ids = (ids[:self.source_max_length] if self.config["truncate"] == "tail" else ids[-self.source_max_length:])
            ids = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + ids + self.suffix_ids)
            self.source_ids.append(torch.tensor(ids, dtype=torch.long))
        if self.set == "train":
            self.target_ids = []
            with tokenizer.as_target_tokenizer():
                target_ids = tokenizer(
                    self.target_text,
                    add_special_tokens=False,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                )["input_ids"]
                for ids in target_ids:
                    ids = (
                        ids[:self.target_max_length]
                        if self.config["truncate"] == "tail" else ids[-self.target_max_length:]
                    )
                    ids = self.tokenizer.build_inputs_with_special_tokens(ids)
                    if self.config["model_name"] in ["bert2bert", "opt", "unilm"]:
                        ids = ids[1:]
                    self.target_ids.append(torch.tensor(ids, dtype=torch.long))


class AbstractCollate:

    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)

        self.is_unilm = (config["model_name"] == "unilm")
        if self.is_unilm:
            if tokenizer.model_max_length < config["src_len"] + self.config["tgt_len"]:
                self.max_len = tokenizer.model_max_length
                self.max_src_len = self.max_len - self.max_len // 2
            else:
                self.max_len = config["src_len"] + self.config["tgt_len"]
                self.max_src_len = config["src_len"]
            self._tril_matrix = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.long))

    @classmethod
    def get_type(cls) -> str:
        return 'pretrain disabled'

    def __call__(self, samples):
        batch = {}
        source_text = []
        source_ids = []
        source_mask = []
        source_length = []
        segment_ids = []
        target_text = []
        position_ids = None
        source_padding_side = (
            "left"
            if self.set != "train" and self.is_casual_model
            else self.tokenizer.padding_side
        )
        source_padding_side = ("left" if self.set != "train" and self.is_casual_model else self.tokenizer.padding_side)

        for index, sample in enumerate(samples):
            source_text.append(sample["source_text"])
            src_id = (
                torch.cat([sample["source_ids"], sample["target_ids"]])
                if self.set == "train" and self.is_casual_model else sample["source_ids"]
            )
            if self.is_unilm:
                n_pad = self.max_len - len(src_id)
                if self.set == "train":
                    n_pad_src = n_pad
                else:
                    n_pad_src = self.max_src_len - len(src_id)
                pad_tensor = torch.tensor([0] * n_pad_src, dtype=torch.long)
                src_id = torch.cat([src_id, pad_tensor])
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                if self.set == "train":
                    second_st, second_end = len(sample["source_ids"]), len(sample["source_ids"]) + len(sample["target_ids"])
                    segment_id = [4] * len(sample["source_ids"]) + [5] * len(sample["target_ids"]) + [0] * n_pad
                else:
                    second_st, second_end = self.max_src_len, self.max_len
                    position_id = []
                    for i in range(len(sample["source_ids"])):
                        position_id.append(i)
                    for i in range(len(sample["source_ids"]), self.max_src_len):
                        position_id.append(0)
                    for i in range(self.max_src_len, self.max_len):
                        position_id.append(i - self.max_src_len + len(sample["source_ids"]))
                    segment_id = [4] * len(sample["source_ids"]) + [5] * n_pad
                    position_id = torch.tensor(position_id, dtype=torch.long)
                    if not index:
                        position_ids = position_id.unsqueeze(0)
                    else:
                        position_ids = torch.cat((position_ids, position_id.unsqueeze(0)), 0)
                input_mask[:, :len(sample["source_ids"])].fill_(1)
                input_mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end - second_st, :second_end - second_st])
                segment_id = torch.tensor(segment_id, dtype=torch.long)
                if not index:
                    segment_ids = segment_id.unsqueeze(0)
                    source_mask = input_mask.unsqueeze(0)
                    source_ids = src_id.unsqueeze(0)
                else:
                    segment_ids = torch.cat((segment_ids, segment_id.unsqueeze(0)), 0)
                    source_mask = torch.cat((source_mask, input_mask.unsqueeze(0)), 0)
                    source_ids = torch.cat((source_ids, src_id.unsqueeze(0)), 0)
            else:
                source_mask.append(torch.ones(len(src_id), dtype=torch.long))
                source_ids.append(src_id)
            source_length.append(len(src_id))
            target_text.append(sample["target_text"])

        batch["source_text"] = source_text
        batch["source_ids"] = _pad_sequence(source_ids, self.tokenizer.pad_token_id, source_padding_side)
        batch["source_mask"] = _pad_sequence(source_mask, 0, source_padding_side)
        if not self.is_unilm:
            batch["source_ids"] = _pad_sequence(
                source_ids, self.tokenizer.pad_token_id, source_padding_side
            )
            batch["source_mask"] = _pad_sequence(source_mask, 0, source_padding_side)
        else:
            batch["source_ids"] = source_ids
            batch["source_mask"] = source_mask
            batch["segment_ids"] = segment_ids
            batch["position_ids"] = position_ids
        batch["source_length"] = torch.tensor(source_length, dtype=torch.long)
        batch["target_text"] = target_text

        if self.set == "train":
            target_ids = []
            for sample in samples:
                tgt_id = (
                    torch.cat([
                        torch.full([len(sample["source_ids"])], -100, dtype=torch.long),
                        sample["target_ids"],
                    ]) if self.is_casual_model and not self.is_unilm else sample["target_ids"]
                )
                target_ids.append(tgt_id)
            batch["target_ids"] = _pad_sequence(
                target_ids, -100, self.tokenizer.padding_side
            )
            if self.is_unilm:
                masked_ids_list = []
                masked_pos_list = []
                masked_weights_list = []
                for index, sample in enumerate(samples):
                    effective_length = len(sample["target_ids"]) - 1
                    n_pred = min(20, max(1, int(round(effective_length*0.2))))
                    # candidate positions of masked tokens
                    cand_pos = []
                    special_pos = set()
                    for i, tk_id in enumerate(batch["source_ids"][index]):
                        if not tk_id:
                            break
                        # only mask tokens_b (target sequence)
                        # we will mask [SEP] as an ending symbol
                        if i >= len(sample["source_ids"]):
                            cand_pos.append(i)
                        else:
                            special_pos.add(i)
                    random.shuffle(cand_pos)

                    masked_pos = set()
                    max_cand_pos = max(cand_pos)
                    for pos in cand_pos:
                        if len(masked_pos) >= n_pred:
                            break
                        if pos in masked_pos:
                            continue

                        st_pos, end_pos = pos, pos + 1

                        for mp in range(st_pos, end_pos):
                            if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                                masked_pos.add(mp)
                            else:
                                break

                    masked_pos = list(masked_pos)
                    if len(masked_pos) > n_pred:
                        random.shuffle(masked_pos)
                        masked_pos = masked_pos[:n_pred]

                    masked_ids = [batch["source_ids"][index][pos].item() for pos in masked_pos]
                    for pos in masked_pos:
                        if random.random() < 0.8:  # 80%
                            batch["source_ids"][index][pos] = self.tokenizer.convert_tokens_to_ids('[MASK]')
                        elif random.random() < 0.5:  # 10%
                            batch["source_ids"][index][pos] = random.randint(1, self.tokenizer.vocab_size-1)
                    # when n_pred < max_pred, we only calculate loss within n_pred
                    masked_weights = [1] * len(masked_ids)

                    # Zero Padding for masked target
                    n_pad = 20 - len(masked_ids)
                    if masked_ids is not None:
                        masked_ids.extend([0] * n_pad)
                    if masked_pos is not None:
                        masked_pos.extend([0] * n_pad)
                    if masked_weights is not None:
                        masked_weights.extend([0] * n_pad)

                    masked_ids_list.append(masked_ids)
                    masked_pos_list.append(masked_pos)
                    masked_weights_list.append(masked_weights)

                batch["masked_ids"] = torch.tensor(masked_ids_list, dtype=torch.long)
                batch["masked_pos"] = torch.tensor(masked_pos_list, dtype=torch.long)
                batch["masked_weights"] = torch.tensor(masked_weights_list, dtype=torch.long)

        return batch
