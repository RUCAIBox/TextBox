# @Time   : 2021/08/01
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

r"""
GPT-2
################################################
Reference:
    Radford et al. "Language models are unsupervised multitask learners".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config


class GPT2Seq(Seq2SeqGenerator):
    r"""GPT-2 is an auto-regressive language model with stacked Transformer decoders.
    """

    def __init__(self, config, dataset):
        super(GPT2Seq, self).__init__(config, dataset)
        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.pretrained_model_path, pad_token='[PAD]', add_prefix_space=True)
        self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path, pad_token_id=self.tokenizer.pad_token_id)
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.task_prefix = config['task_prefix'] if config['task_prefix'] else ''

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        generate_corpus = []

        task_prefix_tokens = self.tokenizer.tokenize(self.task_prefix)
        task_prefix_len = len(task_prefix_tokens)
        for src in source_text:
            input_tokens = self.tokenizer.tokenize(src)[:self.source_max_length - task_prefix_len] + task_prefix_tokens
            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(input_tokens), dtype=torch.long).unsqueeze(0)
            sample_outputs = self.model.generate(
                input_ids,
                num_beams=4,
                max_length=len(input_tokens) + self.target_max_length,
                early_stopping=True,
            )
            generated_text = self.tokenizer.decode(sample_outputs[0][len(input_tokens) + 1:], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
            generated_text = generated_text.lower().split()
            generate_corpus.append(generated_text)

        return generate_corpus

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids = []
        attn_masks = []
        total_length = self.source_max_length + self.target_max_length
        task_prefix_tokens = self.tokenizer.tokenize(self.task_prefix)
        task_prefix_len = len(task_prefix_tokens)
        for src, tgt in zip(source_text, target_text):
            src_tokens = self.tokenizer.tokenize(src)[:self.source_max_length - task_prefix_len]
            tgt_tokens = self.tokenizer.tokenize(tgt)[:self.target_max_length - 1]

            total_tokens = src_tokens + task_prefix_tokens + tgt_tokens + [self.tokenizer.eos_token]
            input_id = self.tokenizer.convert_tokens_to_ids(total_tokens)

            padding_length = total_length - len(input_id)
            attn_mask = [1] * len(input_id) + [0] * padding_length
            input_id = input_id + [self.tokenizer.pad_token_id] * padding_length

            input_ids.append(input_id)
            attn_masks.append(attn_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        attn_masks = torch.tensor(attn_masks, dtype=torch.long).to(self.device)
        labels = input_ids.clone().detach()
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(input_ids, attention_mask=attn_masks, labels=labels)
        return outputs.loss
