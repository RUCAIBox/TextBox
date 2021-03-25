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
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration


class BART(Seq2SeqGenerator):
    r"""BART is a powerful sequence-to-sequence model based on Transformer.
    """

    def __init__(self, config, dataset):
        super(BART, self).__init__(config, dataset)

        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length']

        self.pretrained_model_path = config['pretrained_model_path']
        self.tokenizer = BartTokenizer.from_pretrained(self.pretrained_model_path, add_prefix_space=True)
        self.configuration = BartConfig.from_pretrained(self.pretrained_model_path)

        self.decoder = BartForConditionalGeneration.from_pretrained(
            self.pretrained_model_path, config=self.configuration
        )

        self.padding_token_idx = self.tokenizer.pad_token_id
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
                    sample_outputs = self.decoder.generate(
                        input_ids, num_beams=5, max_length=self.max_target_length, early_stopping=True
                    )
                    generated_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
                    generate_corpus.append(generated_text.lower().split())
        return generate_corpus

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids = []
        attn_masks = []
        for text in source_text:
            sentence = ' '.join(text)
            encoding_dict = self.tokenizer(
                sentence, max_length=self.max_source_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids.append(encoding_dict['input_ids'])
            attn_masks.append(encoding_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attn_masks = torch.cat(attn_masks, dim=0).to(self.device)

        target_ids = []
        decoder_attn_masks = []
        for text in target_text:
            sentence = ' '.join(text)
            decoding_dict = self.tokenizer(
                sentence, max_length=self.max_target_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            target_ids.append(decoding_dict['input_ids'])
            decoder_attn_masks.append(decoding_dict['attention_mask'])
        target_ids = torch.cat(target_ids, dim=0).to(self.device)
        decoder_attn_masks = torch.cat(decoder_attn_masks, dim=0).to(self.device)

        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_attn_masks = decoder_attn_masks[:, :-1].contiguous()
        decoder_target_ids = target_ids[:, 1:].contiguous()

        outputs = self.decoder(
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
        loss = loss.sum(dim=1) / length.float()

        return loss.mean()
