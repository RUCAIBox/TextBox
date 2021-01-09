# @Time   : 2020/11/15
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

r"""
GPT-2
################################################
Reference:
    Radford et al. "Language models are unsupervised multitask".
"""


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import UnconditionalGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


class GPT2(UnconditionalGenerator):
    r"""GPT-2 is an auto-regressive language model with stacked Transformer decoders.

    """

    def __init__(self, config, dataset):
        super(GPT2, self).__init__(config, dataset)

        self.eval_generate_num = config['eval_generate_num']
        
        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model_path,
                                                       bos_token=dataset.sos_token, eos_token=dataset.eos_token,
                                                       pad_token=dataset.padding_token, unk_token=dataset.eos_token)

        self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path)

        self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.padding_token_idx = self.tokenizer.pad_token_id
        self.max_seq_length = config['max_seq_length']

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, eval_data):
        generate_corpus = []
        first_token_ids = set()
        for corpus in eval_data:
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
        attn_masks = []
        for text in text_sequence:
            sentence = ' '.join([self.sos_token] + text + [self.eos_token])
            encoding_dict = self.tokenizer(sentence,
                                           max_length=self.max_seq_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt")
            input_ids.append(encoding_dict['input_ids'])
            attn_masks.append(encoding_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attn_masks = torch.cat(attn_masks, dim=0).to(self.device)

        decoder_input_ids = input_ids[:, :-1].contiguous()
        decoder_labels = input_ids[:, 1:].contiguous()
        attn_masks = attn_masks[:, :-1].contiguous()

        outputs = self.decoder(decoder_input_ids,
                               labels=decoder_labels,
                               attention_mask=attn_masks)

        token_logits = outputs[1]
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_labels.view(-1))
        loss = loss.reshape_as(decoder_labels)

        length = (decoder_labels != self.padding_token_idx).sum(dim=1).float()
        loss = loss.sum(dim=1) / length.float()
        return loss.mean()

    def calculate_nll_test(self, corpus, epoch_idx=-1):
        text_sequence = corpus['target_text']
        input_ids = []
        attn_masks = []
        for text in text_sequence:
            sentence = ' '.join([self.sos_token] + text + [self.eos_token])
            encodings_dict = self.tokenizer(sentence,
                                            max_length=self.max_seq_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt")
            input_ids.append(encodings_dict['input_ids'])
            attn_masks.append(encodings_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attn_masks = torch.cat(attn_masks, dim=0).to(self.device)

        decoder_input_ids = input_ids[:, :-1].contiguous()
        decoder_labels = input_ids[:, 1:].contiguous()
        attn_masks = attn_masks[:, :-1].contiguous()

        outputs = self.decoder(decoder_input_ids,
                               labels=decoder_labels,
                               attention_mask=attn_masks)

        token_logits = outputs[1]
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_labels.view(-1))
        loss = loss.reshape_as(decoder_labels)

        loss = loss.sum(dim=1)
        return loss.mean()
