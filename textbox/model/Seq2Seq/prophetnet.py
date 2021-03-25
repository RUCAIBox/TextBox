# @Time   : 2021/3/15
# @Author : Zhipeng Chen
# @Email  : zhipeng_chen@ruc.edu.cn

r"""
ProphetNet
################################################
Reference:
    Weizhen Qi et al. "ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training" at 2020
"""

import torch
import torch.nn as nn

from textbox.model.abstract_generator import Seq2SeqGenerator
from transformers import ProphetNetConfig, ProphetNetTokenizer, ProphetNetForConditionalGeneration


class ProphetNet(Seq2SeqGenerator):
    r"""ProphetNet is a sequence-to-sequence model based on Transformer.
    """

    def __init__(self, config, dataset):
        super(ProphetNet, self).__init__(config, dataset)

        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length']

        self.pretrained_model_path = config['pretrained_model_path']
        self.config = ProphetNetConfig.from_pretrained(self.pretrained_model_path)
        self.tokenizer = ProphetNetTokenizer.from_pretrained(self.pretrained_model_path, add_prefix_space=True)
        self.model = ProphetNetForConditionalGeneration.from_pretrained(self.pretrained_model_path, config=self.config)

        self.padding_token_idx = self.tokenizer.pad_token_id
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, eval_dataloader):
        generate_corpus = []
        with torch.no_grad():
            sum = 0
            for batch_text in eval_dataloader:
                source_text = batch_text['source_text']
                for text in source_text:
                    text = ' '.join(text)
                    encoding_dict = self.tokenizer(text, return_tensors='pt')
                    input_ids = encoding_dict['input_ids'].to(self.device)
                    output_ids = self.model.generate(input_ids, max_length=self.max_target_length, early_stopping=True)
                    generate_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    generate_corpus.append(generate_text.lower().split())

        return generate_corpus

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        self.batch_size = len(source_text)

        input_ids = []
        input_att = []
        for text in source_text:
            text = ' '.join(text)
            encoding_dict = self.tokenizer(
                text, max_length=self.max_source_length, padding='max_length', return_tensors='pt'
            )
            input_ids.append(encoding_dict['input_ids'])
            input_att.append(encoding_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        input_att = torch.cat(input_att, dim=0).to(self.device)

        target_ids = []
        decoder_input_att = []
        for text in target_text:
            text = ' '.join(text)
            encoding_dict = self.tokenizer(
                text, max_length=self.max_target_length, padding='max_length', return_tensors='pt'
            )
            target_ids.append(encoding_dict['input_ids'])
            decoder_input_att.append(encoding_dict['attention_mask'])
        target_ids = torch.cat(target_ids, dim=0).to(self.device)
        decoder_input_att = torch.cat(decoder_input_att, dim=0).to(self.device)
        decoder_target_ids = target_ids[:, 1:].contiguous()
        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_input_att = decoder_input_att[:, :-1].contiguous()

        outputs = self.model(
            input_ids,
            attention_mask=input_att,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_att,
            use_cache=False
        )

        # token_logits (Torch.Tensor): shape: [batch_size, decoder_sequence_length, vocab_size]
        token_logits = outputs.logits
        loss_main_stream = self.loss(token_logits.reshape(-1, token_logits.size(-1)), decoder_target_ids.reshape(-1))
        loss_main_stream = loss_main_stream.reshape_as(decoder_target_ids)

        # token_logits_ngram (Torch.Tensor): shape: [batch_size, ngram - 1, decoder_sequence_length, vocab_size]
        ngram_decoder_target_ids = torch.cat(
            (target_ids[:, 2:], torch.zeros((self.batch_size, 1), dtype=torch.int64).to(self.device)), dim=1
        )
        token_logits_ngram = outputs.logits_ngram
        loss_predict_stream = self.loss(
            token_logits_ngram.reshape(-1, token_logits_ngram.size(-1)), ngram_decoder_target_ids.reshape(-1)
        )
        loss_predict_stream = loss_predict_stream.reshape_as(ngram_decoder_target_ids)
        loss_predict_stream = loss_predict_stream.sum(dim=0)

        loss = loss_main_stream + loss_predict_stream
        loss = loss.sum(dim=1) / self.max_target_length
        return loss.mean()
