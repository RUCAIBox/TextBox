# @Time   : 2020/11/15
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import UnconditionalGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


class GPT2(UnconditionalGenerator):
    r"""GPT2

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(GPT2, self).__init__(config, dataset)

        if config['gpt2_type'] not in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']:
            raise ValueError("The type of GPT2 model should be given in ['gpt2', 'gpt2-medium', 'gpt2-large', "
                             "'gpt2-xl', 'distilgpt2']")
        else:
            self.gpt2_type = config['gpt2_type']

        self.eval_generate_num = config['eval_generate_num']

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_type,
                                                       bos_token=dataset.sos_token, eos_token=dataset.eos_token,
                                                       pad_token=dataset.padding_token, unk_token=dataset.eos_token)
        self.configuration = GPT2Config.from_pretrained(self.gpt2_type, output_hidden_states=False)

        self.decoder = GPT2LMHeadModel.from_pretrained(self.gpt2_type, config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.padding_token_idx = self.tokenizer.pad_token_id
        self.max_length = config['max_seq_length']

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, eval_data):
        generate_corpus = []

        with torch.no_grad():
            for _ in range(self.eval_generate_num):
                sample_outputs = self.decoder.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=self.max_length,
                    top_p=0.95,
                    num_return_sequences=1
                )
                generated_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
                generate_corpus.append(generated_text.lower().split())
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1):
        text_sequence = corpus['target_text']
        text_idx = []
        attn_mask = []
        for text in text_sequence:
            token_list = [self.sos_token] + text + [self.eos_token]
            encodings_dict = self.tokenizer(' '.join(token_list))
            text_idx.append(encodings_dict['input_ids'])
            attn_mask.append(encodings_dict['attention_mask'])
        max_length = max([len(idx) for idx in text_idx])
        text_idx = [idx + [self.padding_token_idx] * (max_length - len(idx)) for idx in text_idx]
        attn_mask = [mask + [0] * (max_length - len(mask)) for mask in attn_mask]
        text_idx = torch.LongTensor(text_idx).to(self.device)
        attn_mask = torch.LongTensor(attn_mask).to(self.device)

        input_text = text_idx[:, :-1]
        target_text = text_idx[:, 1:]
        attn_mask = attn_mask[:, :-1]
        outputs = self.decoder(input_ids=input_text,
                               labels=input_text,
                               attention_mask=attn_mask)
        token_logits = outputs[1]

        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        length = attn_mask.sum(dim=1) - 1
        loss = loss.sum(dim=1) / length
        return loss.mean()

    def calculate_nll_test(self, corpus, epoch_idx=-1):
        text_sequence = corpus['target_text']
        text_idx = []
        attn_mask = []
        for text in text_sequence:
            token_list = [self.sos_token] + text + [self.eos_token]
            encodings_dict = self.tokenizer(' '.join(token_list))
            text_idx.append(encodings_dict['input_ids'])
            attn_mask.append(encodings_dict['attention_mask'])
        max_length = max([len(idx) for idx in text_idx])
        text_idx = [idx + [self.padding_token_idx] * (max_length - len(idx)) for idx in text_idx]
        attn_mask = [mask + [0] * (max_length - len(mask)) for mask in attn_mask]
        text_idx = torch.LongTensor(text_idx).to(self.device)
        attn_mask = torch.LongTensor(attn_mask).to(self.device)

        input_text = text_idx[:, :-1]
        target_text = text_idx[:, 1:]
        attn_mask = attn_mask[:, :-1]
        outputs = self.decoder(input_ids=input_text,
                               labels=input_text,
                               attention_mask=attn_mask)
        token_logits = outputs[1]

        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        loss = loss.sum(dim=1)
        return loss.mean()

