# @Time   : 2021/3/15
# @Author : Zhuohao Yu
# @Email  : zhuohao@ruc.edu.cn

# UPDATE:
# @Time   : 2022/1/18
# @Author : Wenxun Dai
# @Email  : wxdai@stu.xidian.edu.cn


r"""
T5
################################################
Reference:
    Colin et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" at JMLR 2020.
"""

import torch
import torch.nn as nn
import torch.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class T5(Seq2SeqGenerator):

    def __init__(self, config, dataset):
        super(T5, self).__init__(config, dataset)

        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model_path)
        self.configuration = T5Config.from_pretrained(self.pretrained_model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.task_prefix = config['task_prefix'] if config['task_prefix'] else ''

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        input_ids, attn_masks = self.tokenize_text(source_text, self.source_max_length)

        sample_outputs = self.model.generate(
            input_ids, attention_mask=attn_masks, num_beams=5, max_length=self.target_max_length, early_stopping=True
        )
        generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generate_corpus = [text.lower().split() for text in generated_text]
        return generate_corpus

    def tokenize_text(self, text, max_length, is_target=False):
        texts = [self.task_prefix + t for t in text] if not is_target else text
        encoding_dict = self.tokenizer(
            texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        )

        input_ids = encoding_dict['input_ids'].to(self.device)
        attn_masks = encoding_dict['attention_mask'].to(self.device)
        return input_ids, attn_masks

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids, attn_masks = self.tokenize_text(source_text, self.source_max_length)
        labels, _ = self.tokenize_text(target_text, self.target_max_length, is_target=True)
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(input_ids, attention_mask=attn_masks, labels=labels)

        return outputs.loss
