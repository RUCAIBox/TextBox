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

        self.max_source_length = config['source_max_seq_length']
        self.max_target_length = config['target_max_seq_length']

        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model_path, add_prefix_space=True)
        self.configuration = T5Config.from_pretrained(self.pretrained_model_path)

        self.decoder = T5ForConditionalGeneration.from_pretrained(
            self.pretrained_model_path, config=self.configuration
        )

        self.padding_token_idx = self.tokenizer.pad_token_id
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    @torch.no_grad()
    def generate(self, eval_dataloader):
        generate_corpus = []
        
        for batch_data in eval_dataloader:
            source_text = batch_data["source_text"]
            for text in source_text:
                sequence = "translate German to English: " + ' '.join(text)
                inputs = self.tokenizer(sequence, return_tensors="pt")
                encoded_sequence = inputs['input_ids']
                sample_outputs = self.decoder.generate(
                    encoded_sequence, max_length=self.max_target_length,early_stopping=True
                )
                decoded_sequence = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
                generate_corpus.append(decoded_sequence)
                
        return generate_corpus

    def calculate_ids(self, source_text):
        input_ids = []
        attention_masks = []
        for text in source_text:
            sequence = "translate German to English: " + ' '.join(text)
            inputs = self.tokenizer(sequence, return_tensors="pt", max_length=self.max_source_length, padding="max_length")
            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
        input_ids = torch.cat(input_ids).contiguous().to(self.device)
        attention_masks = torch.cat(attention_masks).contiguous().to(self.device)
        return input_ids, attention_masks

    def calculate_loss(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        print(epoch_idx, self.device)
        input_ids, attention_masks = self.calculate_ids(source_text)
        target_ids, decoder_attention_masks = self.calculate_ids(target_text)

        decoder_input_ids = target_ids[:, :-1].contiguous().to(self.device)
        decoder_attention_masks = decoder_attention_masks[:, :-1].contiguous().to(self.device)
        decoder_target_ids = target_ids[:, 1:].contiguous().to(self.device)

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_masks,
            decoder_input_ids=decoder_input_ids,
            use_cache=False
        )


        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_target_ids.view(-1))
        loss = loss.reshape_as(decoder_target_ids)
        length = (decoder_target_ids != self.padding_token_idx).sum(dim=1).float()
        loss = loss.sum(dim=1) / length.float()

        return loss.mean()