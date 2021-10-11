# @Time   : 2020/12/25
# @Author : Puzhao Xie
# @Email  : xiepuzhao@ruc.edu.cn

r"""
BERT2BERT
################################################
Reference:
    Rothe et al. "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks" in TACL 2020.
"""

import torch
import torch.nn as nn
from textbox.model.abstract_generator import Seq2SeqGenerator
from transformers import BertTokenizer, EncoderDecoderConfig, BertConfig, BertGenerationEncoder, BertGenerationDecoder, \
    EncoderDecoderModel


class BERT2BERT(Seq2SeqGenerator):
    r"""The BertGeneration model is a BERT model that can be leveraged for sequence-to-sequence tasks using EncoderDecoderModel.
    """

    def __init__(self, config, dataset):
        super(BERT2BERT, self).__init__(config, dataset)

        self.sos_token_idx = 101
        self.eos_token_idx = 102
        self.pretrained_model_path = config['pretrained_model_path']

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)

        self.encoder_configure = BertConfig.from_pretrained(self.pretrained_model_path)
        self.decoder_configure = BertConfig.from_pretrained(self.pretrained_model_path)
        self.encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=self.encoder_configure, decoder_config=self.decoder_configure
        )

        self.encoder = BertGenerationEncoder.from_pretrained(
            self.pretrained_model_path, bos_token_id=self.sos_token_idx, eos_token_id=self.eos_token_idx
        )
        self.decoder = BertGenerationDecoder.from_pretrained(
            self.pretrained_model_path,
            bos_token_id=self.sos_token_idx,
            eos_token_id=self.eos_token_idx,
            add_cross_attention=True,
            is_decoder=True
        )
        self.model = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder, config=self.encoder_decoder_config)

        self.padding_token_idx = self.tokenizer.pad_token_id
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        input_ids, attn_masks = self.tokenize_text(source_text)

        sample_outputs = self.model.generate(
            input_ids,
            attention_mask=attn_masks,
            bos_token_id=self.sos_token_idx,
            eos_token_id=self.eos_token_idx,
            num_beams=5,
            max_length=self.target_max_length,
            early_stopping=True
        )
        generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        generate_corpus = [text.lower().split() for text in generated_text]
        return generate_corpus

    def tokenize_text(self, text):
        input_ids = []
        attn_masks = []
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
        target_ids, decoder_attn_masks = self.tokenize_text(target_text)

        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_attn_masks = decoder_attn_masks[:, :-1].contiguous()
        decoder_target_ids = target_ids[:, 1:].contiguous()

        outputs = self.model(
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
        loss = loss.sum(dim=1) / length

        return loss.mean()
