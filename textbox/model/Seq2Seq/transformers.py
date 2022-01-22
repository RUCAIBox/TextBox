import copy

import torch
import torch.nn as nn
import torch.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    BertTokenizer, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
)

MODEL_CLASSES = {
    't5': {
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration
    },
    'bart': {
        'tokenizer': BartTokenizer,
        'model': BartForConditionalGeneration
    },
    'gpt2seq': {
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel
    },
    'bert2bert': {
        'tokenizer': BertTokenizer,
        'model': EncoderDecoderModel
    },

}


class Transformers(Seq2SeqGenerator):
    def __init__(self, config, dataset):
        super(Transformers, self).__init__(config, dataset)

        self.model_name_or_path = config['pretrained_model_path']
        self.model_name = config['model'].lower()

        tokenizer_class = MODEL_CLASSES[self.model_name]['tokenizer']
        model_class = MODEL_CLASSES[self.model_name]['model']

        if self.model_name == 'bert2bert':
            self.model = model_class.from_encoder_decoder_pretrained(self.model_name_or_path, self.model_name_or_path)
        else:
            self.model = model_class.from_pretrained(self.model_name_or_path)

        self.configuration = self.model.config
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path)

        self._check_params()

        self.task_prefix = config['task_prefix'] if config['task_prefix'] else ''
        self.task_infix = config['task_infix'] if config['task_infix'] else ''
        self.task_prefix_ids = self.tokenizer.encode(self.task_prefix, add_special_tokens=False)
        self.task_infix_ids = self.tokenizer.encode(self.task_infix, add_special_tokens=False)

    # def generate(self, batch_data, eval_data):
    #     source_text = batch_data['source_text']
    #
    #     generate_corpus = []
    #
    #     for src in source_text:
    #         src_ids = self.tokenizer.encode(src, add_special_tokens=False)
    #
    #         if self.model_type == 'gpt2':
    #             input_id = src_ids + self.task_infix_ids
    #             max_length = self.max_source_length + len(self.task_infix_ids) + self.max_target_length
    #         else:
    #             input_id = self.task_prefix_ids + src_ids
    #             max_length = self.max_target_length
    #
    #         input_id = torch.tensor(input_id, dtype=torch.long, device=self.device).unsqueeze(1, -1)
    #
    #         outputs = self.model.generate(input_id, num_beams=5, max_length=max_length, early_stopping=True)
    #
    #         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split()
    #
    #         generate_corpus.append(generated_text)
    #
    #     return generate_corpus

    def _generate_default_inputs(self, corpus):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        input_ids = []
        labels = []
        attn_masks = []

        for src, tgt in zip(source_text, target_text):
            src_ids = self.tokenizer.encode(src, add_special_tokens=False)
            tgt_ids = self.tokenizer.encode(tgt, add_special_tokens=False)

            if self.model_name == 'gpt2seq':
                src_ids = src_ids[:self.source_max_length - len(self.task_prefix_ids) - len(self.task_infix_ids)]
                tgt_ids = tgt_ids[:self.target_max_length - len([self.tokenizer.eos_token_id])]
                input_id = self.task_prefix_ids + src_ids + self.task_infix_ids + tgt_ids + [
                    self.tokenizer.eos_token_id]
                label = copy.deepcopy(input_id)

                padding_length = self.source_max_length + self.target_max_length - len(input_id)
                assert padding_length >= 0

                attn_mask = [1] * len(input_id) + [0] * padding_length
                input_id = input_id + [self.tokenizer.pad_token_id] * padding_length
                label = label + [-100] * padding_length

            else:
                src_ids = src_ids[:self.source_max_length - self.tokenizer.num_special_tokens_to_add()
                                  - len(self.task_prefix_ids)]
                tgt_ids = tgt_ids[:self.target_max_length - self.tokenizer.num_special_tokens_to_add()]
                input_id = self.tokenizer.build_inputs_with_special_tokens(self.task_prefix_ids + src_ids)
                label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)

                input_padding_length = self.source_max_length - len(input_id)
                assert input_padding_length >= 0

                attn_mask = [1] * len(input_id) + [0] * input_padding_length
                input_id = input_id + [self.tokenizer.pad_token_id] * input_padding_length

                label_padding_length = self.target_max_length - len(label)
                assert label_padding_length >= 0

                label = label + [-100] * label_padding_length

            assert self.tokenizer.eos_token_id in input_id and self.tokenizer.eos_token_id in label

            input_ids.append(input_id)
            labels.append(label)
            attn_masks.append(attn_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        attn_masks = torch.tensor(attn_masks, dtype=torch.long).to(self.device)

        inputs = {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}
        if self.model_name == 'bert2bert':
            inputs['decoder_input_ids'] = labels.clone().detach()

        return inputs

    def forward(self, corpus, epoch_idx=-1):
        inputs = self._generate_default_inputs(corpus)
        outputs = self.model(**inputs)
        return outputs.loss

    def _check_params(self):
        if self.model_name == 'gpt2seq':
            self.tokenizer.add_prefix_space = True
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.configuration.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif self.model_name == 'bart':
            self.configuration.forced_bos_token_id = self.tokenizer.bos_token_id

        elif self.model_name == 'bert2bert':
            self.configuration.decoder_start_token_id = self.tokenizer.cls_token_id
            self.configuration.eos_token_id = self.tokenizer.sep_token_id
            self.tokenizer.eos_token = self.tokenizer.sep_token
