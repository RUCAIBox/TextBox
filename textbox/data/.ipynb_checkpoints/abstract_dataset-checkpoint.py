# import os
# import torch, warnings
# from torch.utils.data import DataLoader, Dataset
# from textbox import CLM_MODELS
# from typing import List
# from torch.nn.utils.rnn import pad_sequence
# from textbox.data.misc import load_data, _pad_sequence


# class AbstractDataset(Dataset):
#     def __init__(self, config, set):
#         super().__init__()
#         self.config = config
#         self.quick_test = config["quick_test"]
#         if isinstance(self.quick_test, bool):
#             self.quick_test = 32 if self.quick_test else 0
#         self.set = set
#         source_filename = os.path.join(config["data_path"], f"{set}.src")
#         target_filename = os.path.join(config["data_path"], f"{set}.tgt")
#         self.source_text = load_data(source_filename, max_length=self.quick_test)
#         self.target_text = load_data(target_filename, max_length=self.quick_test)
#         self.source_length = self.config["src_len"]
#         self.target_length = self.config["tgt_len"]
#         if set == "train":
#             assert len(self.source_text) == len(self.target_text)

#     def __len__(self):
#         return len(self.source_text)

#     def __getitem__(self, idx):
#         sample = {
#             "source_text": self.source_text[idx],
#             "source_ids": self.source_ids[idx],
#             "target_text": self.target_text[idx],
#         }
#         if self.set == "train":
#             sample.update(
#                 {
#                     "target_ids": self.target_ids[idx],
#                 }
#             )
#         return sample

#     def _init_process(self):
#         if self.source_length > self.tokenizer.model_max_length:
#             warnings.warn(
#                 f"The max length of source text {self.source_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and will be set to {self.tokenizer.model_max_length}."
#             )
#             self.source_length = self.tokenizer.model_max_length

#         if self.target_length > self.tokenizer.model_max_length:
#             warnings.warn(
#                 f"The max length of target text {self.target_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and will be set to {self.tokenizer.model_max_length}."
#             )
#             self.target_length = self.tokenizer.model_max_length

#         if (
#             self.config["efficient_methods"]
#             and "prompt-tuning" in self.config["efficient_methods"]
#         ):
#             prompt_length = self.config["efficient_kwargs"]["prompt_length"]
#             if self.config["model_name"] in CLM_MODELS:
#                 if (
#                     self.source_length + self.target_length + prompt_length
#                     > self.tokenizer.model_max_length
#                 ):
#                     warnings.warn(
#                         f"The length of source text {self.source_length}, target text {self.target_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length - self.target_length}."
#                     )
#                     self.source_length = (
#                         self.tokenizer.model_max_length
#                         - prompt_length
#                         - self.target_length
#                     )
#             elif self.source_length + prompt_length > self.tokenizer.model_max_length:
#                 warnings.warn(
#                     f"The length of source text {self.source_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length}."
#                 )
#                 self.source_length = self.tokenizer.model_max_length - prompt_length

#     def _process_prompt(self):
#         prefix = self.config["prefix_prompt"] or ""
#         suffix = self.config["suffix_prompt"] or ""

#         self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
#         self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

#         self.source_max_length = (
#             self.source_length
#             - self.tokenizer.num_special_tokens_to_add()
#             - len(self.prefix_ids)
#             - len(self.suffix_ids)
#         )
#         self.target_max_length = (
#             self.target_length - self.tokenizer.num_special_tokens_to_add()
#         )

#         if self.config["model_name"] in ["bert2bert", "opt"]:
#             self.target_max_length += 1

#     def tokenize(self, tokenizer):
#         self.tokenizer = tokenizer
#         self._init_process()
#         self._process_prompt()
#         self.source_ids = []
#         source_ids = tokenizer(
#             self.source_text,
#             add_special_tokens=False,
#             return_token_type_ids=False,
#             return_attention_mask=False,
#         )["input_ids"]
#         for ids in source_ids:
#             ids = (
#                 ids[: self.source_max_length]
#                 if self.config["truncate"] == "tail"
#                 else ids[-self.source_max_length :]
#             )
#             ids = self.tokenizer.build_inputs_with_special_tokens(
#                 self.prefix_ids + ids + self.suffix_ids
#             )
#             self.source_ids.append(torch.tensor(ids, dtype=torch.long))
#         if self.set == "train":
#             self.target_ids = []
#             with tokenizer.as_target_tokenizer():
#                 target_ids = tokenizer(
#                     self.target_text,
#                     add_special_tokens=False,
#                     return_token_type_ids=False,
#                     return_attention_mask=False,
#                 )["input_ids"]
#                 for ids in target_ids:
#                     ids = (
#                         ids[: self.target_max_length]
#                         if self.config["truncate"] == "tail"
#                         else ids[-self.target_max_length :]
#                     )
#                     ids = self.tokenizer.build_inputs_with_special_tokens(ids)
#                     if self.config["model_name"] in ["bert2bert", "opt"]:
#                         ids = ids[1:]
#                     self.target_ids.append(torch.tensor(ids, dtype=torch.long))


# class AbstractCollate:
#     def __init__(self, config, tokenizer, set):
#         self.config = config
#         self.tokenizer = tokenizer
#         self.set = set
#         self.is_casual_model = bool(config["model_name"] in CLM_MODELS)
    
#     @classmethod
#     def get_type(cls) -> str:
#         return 'pretrain disabled'

#     def __call__(self, samples):
#         batch = {}
#         source_text = []
#         source_ids = []
#         source_mask = []
#         source_length = []
#         target_text = []
#         source_padding_side = (
#             "left"
#             if self.set != "train" and self.is_casual_model
#             else self.tokenizer.padding_side
#         )

#         for sample in samples:
#             source_text.append(sample["source_text"])
#             src_id = (
#                 torch.cat([sample["source_ids"], sample["target_ids"]])
#                 if self.set == "train" and self.is_casual_model
#                 else sample["source_ids"]
#             )
#             source_ids.append(src_id)
#             source_mask.append(torch.ones(len(src_id), dtype=torch.long))
#             source_length.append(len(src_id))
#             target_text.append(sample["target_text"])

#         batch["source_text"] = source_text
#         batch["source_ids"] = _pad_sequence(
#             source_ids, self.tokenizer.pad_token_id, source_padding_side
#         )
#         batch["source_mask"] = _pad_sequence(source_mask, 0, source_padding_side)
#         batch["source_length"] = torch.tensor(source_length, dtype=torch.long)
#         batch["target_text"] = target_text

#         if self.set == "train":
#             target_ids = []
#             for sample in samples:
#                 tgt_id = (
#                     torch.cat(
#                         [
#                             torch.full(
#                                 [len(sample["source_ids"])], -100, dtype=torch.long
#                             ),
#                             sample["target_ids"],
#                         ]
#                     )
#                     if self.is_casual_model
#                     else sample["target_ids"]
#                 )
#                 target_ids.append(tgt_id)
#             batch["target_ids"] = _pad_sequence(
#                 target_ids, -100, self.tokenizer.padding_side
#             )
#         return batch
import os
import torch, warnings
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
        # 获得源和目标文本名称
        self.source_text = load_data(source_filename, max_length=self.quick_test)
        self.target_text = load_data(target_filename, max_length=self.quick_test)
        # 获得文本
        self.source_length = self.config["src_len"]
        self.target_length = self.config["tgt_len"]
        if set == "train":
            assert len(self.source_text) == len(self.target_text)

    def __len__(self):
        return len(self.source_text)
        # 获得原文本长度

    def __getitem__(self, idx):
        sample = {
            "source_text": self.source_text[idx],
            "source_ids": self.source_ids[idx],
            "target_text": self.target_text[idx],
        }
        if self.set == "train":
            sample.update(
                {
                    "target_ids": self.target_ids[idx],
                }
            )
            # 如果是train，添加键值对
        return sample

    def _init_process(self):
        # 对最大长度进行裁剪
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
        # 如果本身的长度大于模型长度，裁剪
        # if (
        #     self.config["efficient_methods"]
        #     and "prompt-tuning" in self.config["efficient_methods"]
        # ):
        # # 如果是prompt-tuning，还需要对长度进行裁剪
        #     prompt_length = self.config["efficient_kwargs"]["prompt_length"]
        #     # if self.config["model_name"] in CLM_MODELS:
        #     #     # 因果语言模型
        #     #     if (
        #     #         self.source_length + self.target_length + prompt_length
        #     #         > self.tokenizer.model_max_length
        #     #     ):
        #     #         warnings.warn(
        #     #             f"The length of source text {self.source_length}, target text {self.target_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length - self.target_length}."
        #     #         )
        #     #         self.source_length = (
        #     #             self.tokenizer.model_max_length
        #     #             - prompt_length
        #     #             - self.target_length
        #     #         )
        #     # 不考虑因果语言模型
        #     # elif self.source_length + prompt_length > self.tokenizer.model_max_length:
        #     if self.source_length + prompt_length > self.tokenizer.model_max_length:
            
        #         warnings.warn(
        #             f"The length of source text {self.source_length} and prompt {prompt_length} exceeds the max length {self.tokenizer.model_max_length} of {self.config['model']} model, and the max length of source sentence will be set to {self.tokenizer.model_max_length - prompt_length}."
        #         )
        #         self.source_length = self.tokenizer.model_max_length - prompt_length

    def _process_prompt(self):
        # 处理额外提示过后的长度
        prefix = self.config["prefix_prompt"] or ""
        suffix = self.config["suffix_prompt"] or ""
        # 在这条赋值语句中的 or 的含义是判断 b 和 c 中不为 None 的一个赋值给 a，两个都不为 None 则选择前面的赋值给 a。

        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        # 将prefix或者suffix进行编码

        self.source_max_length = (
            self.source_length
            - self.tokenizer.num_special_tokens_to_add()
            - len(self.prefix_ids)
            - len(self.suffix_ids)
        )
        # 重新确定最大长度
        self.target_max_length = (
            self.target_length - self.tokenizer.num_special_tokens_to_add()
        )

        # if self.config["model_name"] in ["bert2bert", "opt"]:
        #     self.target_max_length += 1

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
        # tokenize source文本并获得id
        for ids in source_ids:
            ids = (
                ids[: self.source_max_length]
                if self.config["truncate"] == "tail"
                else ids[-self.source_max_length :]
            )
            # 按最大长度截断
            ids = self.tokenizer.build_inputs_with_special_tokens(
                self.prefix_ids + ids + self.suffix_ids
            )
            # 增加特殊的tokn
            self.source_ids.append(torch.tensor(ids, dtype=torch.long))
            # 转化成tensor
        if self.set == "train":
            # 训练
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
                        ids[: self.target_max_length]
                        if self.config["truncate"] == "tail"
                        else ids[-self.target_max_length :]
                    )
                    ids = self.tokenizer.build_inputs_with_special_tokens(ids)
                    # if self.config["model_name"] in ["bert2bert", "opt"]:
                    #     ids = ids[1:]
                    self.target_ids.append(torch.tensor(ids, dtype=torch.long))
                    # 获得target的tensor
        if self.set != "train" and isinstance(self.target_text[0], str):
            tmp_target_ids = [tokenizer.encode(sentence) for sentence in self.target_text]
            self.target_tokens = [
                tokenizer.decode(sentence, skip_special_tokens=True)
                for sentence in tmp_target_ids
            ]


class AbstractCollate:
    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)
    
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
        source_padding_side = (
            # "left"
            # if self.set != "train" and self.is_casual_model
            # else self.tokenizer.padding_side
            self.tokenizer.padding_side
            # 不是casual LM
        )
        # padding的位置

        for sample in samples:
            source_text.append(sample["source_text"])
            src_id = (
                torch.cat([sample["source_ids"], sample["target_ids"]])
                if self.set == "train" and self.is_casual_model
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
        # 获取每个batch的内容

        if self.set == "train":
            target_ids = []
            for sample in samples:
                tgt_id = (
                    torch.cat(
                        [
                            torch.full(
                                [len(sample["source_ids"])], -100, dtype=torch.long
                            ),
                            sample["target_ids"],
                        ]
                    )
                    if self.is_casual_model
                    else sample["target_ids"]
                )
                target_ids.append(tgt_id)
            batch["target_ids"] = _pad_sequence(
                target_ids, -100, self.tokenizer.padding_side
            )
            # pad的部分用-100
        return batch
