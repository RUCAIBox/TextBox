# @Time   : 2020/11/16
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import ConditionalGenerator
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration


class BART(ConditionalGenerator):
    r"""BART is a powerful sequence-to-sequence model based on Transformer.

    Reference:
        https://arxiv.org/abs/1910.13461
    """
    def __init__(self, config, dataset):
        super(BART, self).__init__(config, dataset)

        self.tokenizer = BartTokenizer.from_pretrained('pretrained_model/bart_base',
                                                       bos_token=dataset.sos_token,
                                                       eos_token=dataset.eos_token,
                                                       pad_token=dataset.padding_token,
                                                       unk_token=dataset.unknown_token,
                                                       add_prefix_space=True)
        self.configuration = BartConfig.from_pretrained('pretrained_model/bart_base')

        self.decoder = BartForConditionalGeneration.from_pretrained("pretrained_model/bart_base",
                                                                    config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.padding_token_idx = self.tokenizer.pad_token_id
        self.max_source_length = config['source_max_seq_length']
        self.max_target_length = config['target_max_seq_length']

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, eval_dataloader):
        generate_corpus = []
        with torch.no_grad():
            for batch_data in eval_dataloader:
                source_text = batch_data["source_text"]
                for text in source_text:
                    sentence = ' '.join(text)
                    encoding_dict = self.tokenizer(sentence, return_tensors="pt")
                    input_ids = encoding_dict['input_ids'].to(self.device)
                    sample_outputs = self.decoder.generate(input_ids,
                                                           num_beams=4,
                                                           max_length=self.max_target_length,
                                                           early_stopping=True)
                    generated_text = [self.tokenizer.decode(sample, skip_special_tokens=True) for sample in
                                      sample_outputs]
                    generated_text = [text.lower().split() for text in generated_text]
                    generate_corpus.extend(generated_text)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids = []
        attn_masks = []
        for text in source_text:
            sentence = ' '.join(text)
            encoding_dict = self.tokenizer(sentence,
                                           max_length=self.max_source_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt")
            input_ids.append(encoding_dict['input_ids'])
            attn_masks.append(encoding_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attn_masks = torch.cat(attn_masks, dim=0).to(self.device)

        target_ids = []
        for text in target_text:
            sentence = ' '.join([self.sos_token] + text + [self.eos_token])
            encoding_dict = self.tokenizer(sentence,
                                           max_length=self.max_target_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt")
            target_ids.append(encoding_dict['input_ids'])
        target_ids = torch.cat(target_ids, dim=0).to(self.device)

        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_labels = target_ids[:, 1:].contiguous()

        outputs = self.decoder(input_ids,
                               attention_mask=attn_masks,
                               decoder_input_ids=decoder_input_ids,
                               labels=decoder_labels)

        token_logits = outputs[1]
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_labels.view(-1))
        loss = loss.reshape_as(decoder_labels)

        length = (decoder_labels != self.padding_token_idx).sum(dim=1).float()
        loss = loss.sum(dim=1) / length

        return loss.mean()
