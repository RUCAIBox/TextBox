# @Time   : 2021/3/15
# @Author : Zhuohao Yu
# @Email  : zhuohao@ruc.edu.cn

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

        self.max_source_length = dataset.max_source_length
        self.max_target_length = dataset.max_target_length

        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model_path)
        self.configuration = T5Config.from_pretrained(self.pretrained_model_path)

        self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model_path, config=self.configuration)

        self.padding_token_idx = self.tokenizer.pad_token_id
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')
        if config['task_type'] == "summarization":
            self.t5_task_text = "summarize: "
        elif config['task_type'] == "translation":
            self.t5_task_text = "translate German to English: "
        else:
            raise NotImplementedError("Only summarization and translation are supported.")

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        input_ids, attn_masks = self.tokenize_text(source_text)

        sample_outputs = self.model.generate(
            input_ids, attention_mask=attn_masks, num_beams=5, max_length=self.max_target_length, early_stopping=True
        )
        generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        generate_corpus = [text.lower().split() for text in generated_text]
        return generate_corpus

    def tokenize_text(self, text, is_target=False):
        input_ids = []
        attn_masks = []
        texts = [(self.t5_task_text if not is_target else '') + ' '.join(t) for t in text]
        encoding_dict = self.tokenizer(
            texts, max_length=self.max_source_length, padding=True, truncation=True, return_tensors="pt"
        )

        input_ids = encoding_dict['input_ids'].to(self.device)
        attn_masks = encoding_dict['attention_mask'].to(self.device)

        return input_ids, attn_masks

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids, attn_masks = self.tokenize_text(source_text)
        target_ids, decoder_attn_masks = self.tokenize_text(target_text, is_target=True)

        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_attn_masks = decoder_attn_masks[:, :-1].contiguous()
        decoder_target_ids = target_ids[:, 1:].contiguous()

        outputs = self.model(
            input_ids,
            attention_mask=attn_masks,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attn_masks,
            use_cache=False
        )

        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_target_ids.view(-1))
        loss = loss.reshape_as(decoder_target_ids)

        length = (decoder_target_ids != self.padding_token_idx).sum(dim=1).float()
        loss = loss.sum(dim=1) / length

        return loss.mean()
