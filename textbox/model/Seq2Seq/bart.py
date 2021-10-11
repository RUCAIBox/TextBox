# @Time   : 2020/11/16
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

r"""
BART
################################################
Reference:
    Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
    Generation, Translation, and Comprehensio" at ACL 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator
from transformers import BartTokenizerFast, BartConfig, BartForConditionalGeneration


class BART(Seq2SeqGenerator):
    r"""BART is a powerful sequence-to-sequence model based on Transformer.
    """

    def __init__(self, config, dataset):
        super(BART, self).__init__(config, dataset)
        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = BartTokenizerFast.from_pretrained(self.pretrained_model_path)
        self.configuration = BartConfig.from_pretrained(self.pretrained_model_path)

        self.model = BartForConditionalGeneration.from_pretrained(self.pretrained_model_path, config=self.configuration)

        self.padding_token_idx = self.tokenizer.pad_token_id
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='mean')

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        input_ids, attn_masks = self.tokenize_text(source_text)

        sample_outputs = self.model.generate(
            input_ids, attention_mask=attn_masks, num_beams=5, max_length=self.target_max_length, early_stopping=True
        )
        generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        generate_corpus = [text.lower().split() for text in generated_text]
        return generate_corpus

    def tokenize_text(self, text):
        texts = [' '.join(t) for t in text]
        encoding_dict = self.tokenizer(
            texts, max_length=self.source_max_length, padding=True, truncation=True, return_tensors="pt"
        )

        input_ids = encoding_dict['input_ids'].to(self.device)
        attn_masks = encoding_dict['attention_mask'].to(self.device)

        return input_ids, attn_masks

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids, attn_masks = self.tokenize_text(source_text)
        decoder_ids, decoder_attn_masks = self.tokenize_text(target_text)

        decoder_input_ids = decoder_ids[:, :-1].contiguous()
        decoder_attn_masks = decoder_attn_masks[:, :-1].contiguous()
        labels = decoder_ids[:, 1:].contiguous()
        labels = torch.where(labels != self.padding_token_idx, labels, -100)

        outputs = self.model(
            input_ids,
            attention_mask=attn_masks,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attn_masks,
            use_cache=False
        )

        token_logits = outputs.logits
        loss = nn.CrossEntropyLoss()(token_logits.view(-1, token_logits.size(-1)), labels.view(-1))

        return loss
