# @Time   : 2020/12/25
# @Author : Puzhao Xie
# @Email  : xiepuzhao@ruc.edu.cn

r"""
XLNet
################################################
Reference:
    Yang et al. "XLNet: Generalized Autoregressive Pretraining for Language Understanding" in NIPS 2019.
"""


import random
import torch
import torch.nn as nn

from textbox.model.abstract_generator import UnconditionalGenerator
from transformers import XLNetLMHeadModel, XLNetTokenizer, XLNetConfig


class XLNet(UnconditionalGenerator):
    r""" XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn
    bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence
    factorization order.
    """

    def __init__(self, config, dataset):
        super(XLNet, self).__init__(config, dataset)

        self.eval_generate_num = config['eval_generate_num']

        self.tokenizer = XLNetTokenizer.from_pretrained(
            'xlnet-base-cased',
            bos_token=dataset.sos_token, eos_token=dataset.eos_token,
            pad_token=dataset.padding_token, unk_token=dataset.eos_token)

        self.configuration = XLNetConfig.from_pretrained('xlnet-base-cased')

        self.decoder = XLNetLMHeadModel.from_pretrained('xlnet-base-cased',
            config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.mask_token = '<mask>'
        self.padding_token_idx = self.tokenizer.pad_token_id
        self.max_seq_length = config['max_seq_length']
        self.device = config["device"]

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, eval_data):
        generate_corpus = []
        first_token_ids = set()
        for i, corpus in enumerate(eval_data):
            text_sequence = corpus["target_text"]
            for text in text_sequence:
                sentence = ' '.join(text)
                encoding_dict = self.tokenizer(sentence)
                first_token_ids.add(encoding_dict['input_ids'][0])
        first_token_ids = list(first_token_ids)

        with torch.no_grad():
            for _ in range(self.eval_generate_num):
                sample_outputs = self.decoder.generate(
                    bos_token_id=random.choice(first_token_ids),
                    do_sample=True,
                    top_k=50,
                    max_length=self.max_seq_length,
                    top_p=0.95,
                    num_return_sequences=1
                )
                generated_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
                generate_corpus.append(generated_text.lower().split())
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1):
        text_sequence = corpus['target_text']
        input_ids = []
        for text in text_sequence:
            sentence = ' '.join([self.sos_token] + text + [self.eos_token])
            encoding_dict = self.tokenizer(sentence,
                                           max_length=self.max_seq_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt",
                                           add_special_tokens=False,
                                           )
            input_ids.append(encoding_dict.input_ids)
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        decoder_input_ids = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        perm_mask = torch.ones((input_ids.shape[0],
                                decoder_input_ids.shape[1],
                                decoder_input_ids.shape[1]),
                               dtype=torch.float).to(self.device)
        for t_index in range(self.max_seq_length-1):
            perm_mask[:, -t_index, -(self.max_seq_length-1):-t_index] = 0.0
        perm_mask = perm_mask.contiguous()
        target_mapping = torch.zeros((input_ids.shape[0], decoder_input_ids.shape[1], decoder_input_ids.shape[1]),
                                     dtype=torch.float).to(self.device)
        for index in range(self.max_seq_length-1):
            target_mapping[:, index, index] = 1.0
        target_mapping = target_mapping.contiguous()
        outputs = self.decoder(decoder_input_ids,
                               labels=labels, perm_mask=perm_mask, target_mapping=target_mapping)

        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), labels.view(-1))
        loss = loss.reshape_as(labels)

        length = (labels != self.padding_token_idx).sum(dim=1).float()
        loss = loss.sum(dim=1) / length.float()
        return loss.mean()

    def calculate_nll_test(self, corpus, epoch_idx=-1):
        text_sequence = corpus['target_text']
        input_ids = []
        for text in text_sequence:
            sentence = ' '.join([self.sos_token] + text + [self.eos_token])
            encoding_dict = self.tokenizer(sentence,
                                           max_length=self.max_seq_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt",
                                           add_special_tokens=False,
                                           )
            input_ids.append(encoding_dict.input_ids)
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        decoder_input_ids = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        perm_mask = torch.ones((input_ids.shape[0],
                                decoder_input_ids.shape[1],
                                decoder_input_ids.shape[1]),
                               dtype=torch.float).to(self.device)
        for t_index in range(self.max_seq_length - 1):
            perm_mask[:, -t_index, -(self.max_seq_length - 1):-t_index] = 0.0
        perm_mask = perm_mask.contiguous()
        target_mapping = torch.zeros((input_ids.shape[0], decoder_input_ids.shape[1], decoder_input_ids.shape[1]),
                                     dtype=torch.float).to(
            self.device)
        for index in range(self.max_seq_length - 1):
            target_mapping[:, index, index] = 1.0
        target_mapping = target_mapping.contiguous()
        outputs = self.decoder(decoder_input_ids,
                               labels=labels, perm_mask=perm_mask, target_mapping=target_mapping)

        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), labels.view(-1))
        loss = loss.reshape_as(labels)
        loss = loss.sum(dim=1)
        return loss.mean()
