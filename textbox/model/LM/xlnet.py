# @Time   : 2020/12/25
# @Author : Puzhao Xie
# @Email  : xiepuzhao@ruc.edu.cn

r"""
XLNet
################################################
Reference:
    Yang et al. "XLNet: Generalized Autoregressive Pretraining for Language Understanding" in NIPS 2019.
"""

import torch
import torch.nn as nn

from textbox.model.abstract_generator import UnconditionalGenerator
from transformers import XLNetLMHeadModel, XLNetTokenizer, XLNetConfig
from math import ceil


class XLNet(UnconditionalGenerator):
    r""" XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn
    bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence
    factorization order.
    """

    def __init__(self, config, dataset):
        super(XLNet, self).__init__(config, dataset)

        self.eval_generate_num = config['eval_generate_num']

        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = XLNetTokenizer.from_pretrained(
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

        self.configuration = XLNetConfig.from_pretrained(
            self.pretrained_model_path,
            bos_token_id=self.sos_token_idx,
            eos_token_id=self.eos_token_idx,
            pad_token_id=self.padding_token_idx
        )

        self.decoder = XLNetLMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, eval_data):
        generate_corpus = []
        batch_num = ceil(self.eval_generate_num / eval_data.batch_size)
        for _ in range(batch_num):
            sample_outputs = self.decoder.generate(
                bos_token_id=self.sos_token_idx,
                do_sample=True,
                max_length=self.max_seq_length,
                num_return_sequences=eval_data.batch_size
            )
            generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
            generate_corpus.extend([text.lower().split() for text in generated_text])
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1, nll_test=False):
        text_sequence = corpus['target_text']
        input_ids = []
        attn_masks = []
        for text in text_sequence:
            sentence = ' '.join([self.sos_token] + text + [self.eos_token])
            encoding_dict = self.tokenizer(
                sentence,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
            input_ids.append(encoding_dict.input_ids)
            attn_masks.append(encoding_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attn_masks = torch.cat(attn_masks, dim=0).to(self.device)

        decoder_target_ids = input_ids[:, 1:].contiguous()

        perm_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]).to(self.device)
        perm_mask = perm_mask.triu(diagonal=1)

        target_ones = torch.ones(input_ids.shape[1] - 1).to(self.device)
        target_ones = target_ones.diag(1)[:-1]
        target_mapping = target_ones.expand(input_ids.shape[0], -1, -1)

        outputs = self.decoder(input_ids, attention_mask=attn_masks, perm_mask=perm_mask, target_mapping=target_mapping)

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
