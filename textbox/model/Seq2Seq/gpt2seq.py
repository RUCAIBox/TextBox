# @Time   : 2021/08/01
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

r"""
GPT-2
################################################
Reference:
    Radford et al. "Language models are unsupervised multitask".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from math import ceil


class GPT2Seq(Seq2SeqGenerator):
    r"""GPT-2 is an auto-regressive language model with stacked Transformer decoders.

    """

    def __init__(self, config, dataset):
        super(GPT2Seq, self).__init__(config, dataset)

        self.max_source_length = dataset.max_source_length
        self.max_target_length = dataset.max_target_length

        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.pretrained_model_path, pad_token='[PAD]')
        
        self.eos_token = self.tokenizer.eos_token
        self.padding_token_idx = self.tokenizer.pad_token_id

        self.configuration = GPT2Config.from_pretrained(
            self.pretrained_model_path,
            pad_token_id=self.padding_token_idx
        )

        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if config['task_type'] == "summarization":
            self.task_text = "TL;DR:"
        elif config['task_type'] == "translation":
            self.task_text = "story:"
        elif config['task_type'] == "multi_dialog":
            self.task_text = "question:"
        else:
            raise NotImplementedError("Only summarization and translation are supported.")

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        generate_corpus = []
        for src in source_text:
            input_ids = self.tokenize_text(src, self.task_text, self.max_source_length).unsqueeze(0)

            sample_outputs = self.model.generate(
                input_ids,
                num_beams=4, max_length=input_ids.size(1) + self.max_target_length, early_stopping=True,
            )
            generated_text = self.tokenizer.decode(sample_outputs[0][input_ids.size(1) + 1:], skip_special_tokens=True)
            generated_text = generated_text.split()
            generate_corpus.append(generated_text)
        return generate_corpus

    def tokenize_text(self, text, suff_text, max_length):
        suff_dict = self.tokenizer(' ' + suff_text, return_tensors="pt")
        suff_ids = suff_dict['input_ids'].to(self.device)[0]

        texts = ' '.join(text)
        encoding_dict = self.tokenizer(texts, max_length=max_length-suff_ids.size(0), truncation=True, return_tensors="pt")
        input_ids = encoding_dict['input_ids'].to(self.device)[0]

        input_ids = torch.cat((input_ids, suff_ids)).long()
        return input_ids

    def forward(self, corpus, epoch_idx=-1, nll_test=False):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids = []
        src_length = []
        for src, tgt in zip(source_text, target_text):
            src_ids = self.tokenize_text(src, self.task_text, self.max_source_length)
            tgt_ids = self.tokenize_text(tgt, self.eos_token, self.max_target_length)
            input_id = torch.cat((src_ids, tgt_ids))
            
            src_length.append(src_ids.size(0))
            input_ids.append(input_id)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.padding_token_idx)
        attn_masks = input_ids != self.padding_token_idx
        loss_masks = attn_masks.clone().detach()
        for i, l in enumerate(src_length):
            loss_masks[i][:l] = 0

        decoder_input_ids = input_ids[:, :-1].contiguous()
        decoder_target_ids = input_ids[:, 1:].contiguous()
        attn_masks = attn_masks[:, :-1].contiguous()
        loss_masks = loss_masks[:, :-1].contiguous()

        outputs = self.model(decoder_input_ids, attention_mask=attn_masks, use_cache=False)

        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_target_ids.view(-1))
        loss = loss.reshape_as(decoder_target_ids) * loss_masks

        length = ((decoder_target_ids != self.padding_token_idx) * loss_masks).sum(dim=1).float()
        loss = loss.sum(dim=1) / length
        return loss.mean()
