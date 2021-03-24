# @Time   : 2020/11/15
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

r"""
GPT-2
################################################
Reference:
    Radford et al. "Language models are unsupervised multitask".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import UnconditionalGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from math import ceil


class GPT2(UnconditionalGenerator):
    r"""GPT-2 is an auto-regressive language model with stacked Transformer decoders.

    """

    def __init__(self, config, dataset):
        super(GPT2, self).__init__(config, dataset)

        self.eval_generate_num = config['eval_generate_num']

        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.pretrained_model_path,
            bos_token=dataset.sos_token,
            eos_token=dataset.eos_token,
            pad_token=dataset.padding_token
        )

        self.sos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.sos_token_idx = self.tokenizer.bos_token_id
        self.eos_token_idx = self.tokenizer.eos_token_id
        self.padding_token_idx = self.tokenizer.pad_token_id
        self.max_seq_length = config['max_seq_length']

        self.configuration = GPT2Config.from_pretrained(
            self.pretrained_model_path,
            bos_token_id=self.sos_token_idx,
            eos_token_id=self.eos_token_idx,
            pad_token_id=self.padding_token_idx
        )

        self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, batch_data, eval_data):
        generate_corpus = []
        batch_num = ceil(self.eval_generate_num / eval_data.batch_size)
        sample_outputs = self.decoder.generate(
            bos_token_id=self.sos_token_idx,
            do_sample=True,
            max_length=self.max_seq_length,
            num_return_sequences=eval_data.batch_size
        )
        generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        generate_corpus.extend([text.lower().split() for text in generated_text])
        return generate_corpus

    def forward(self, corpus, epoch_idx=-1, nll_test=False):
        text_sequence = corpus['target_text']
        input_ids = []
        attn_masks = []
        for text in text_sequence:
            sentence = ' '.join([self.sos_token] + text + [self.eos_token])
            encoding_dict = self.tokenizer(
                sentence, max_length=self.max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids.append(encoding_dict['input_ids'])
            attn_masks.append(encoding_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attn_masks = torch.cat(attn_masks, dim=0).to(self.device)

        decoder_input_ids = input_ids[:, :-1].contiguous()
        decoder_target_ids = input_ids[:, 1:].contiguous()
        attn_masks = attn_masks[:, :-1].contiguous()

        outputs = self.decoder(decoder_input_ids, attention_mask=attn_masks, use_cache=False)

        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_target_ids.view(-1))
        loss = loss.reshape_as(decoder_target_ids)

        if nll_test:
            loss = loss.sum(dim=1)
        else:
            length = (decoder_target_ids != self.padding_token_idx).sum(dim=1).float()
            loss = loss.sum(dim=1) / length.float()
        return loss.mean()

    def calculate_nll_test(self, corpus, epoch_idx=-1):
        return self.calculate_loss(corpus, epoch_idx, True)
