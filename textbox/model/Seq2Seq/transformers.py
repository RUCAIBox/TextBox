import copy

import torch
import torch.nn as nn
import torch.functional as F

from textbox.model.abstract_generator import Seq2SeqGenerator

from transformers import (
    T5Config, T5Tokenizer, T5ForConditionalGeneration,
    BartConfig, BartTokenizer, BartForConditionalGeneration
)

MODEL_CLASSES = {
    'T5': {
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration
    },

}


class Transformers(Seq2SeqGenerator):
    def __init__(self, config, dataset):
        super(Transformers, self).__init__(config, dataset)

        self.pretrained_model_path = config['pretrained_model_path']
        config_class = MODEL_CLASSES[self.config['model']]['config']
        tokenizer_class = MODEL_CLASSES[self.config['model']]['tokenizer']
        model_class = MODEL_CLASSES[self.config['model']]['model']

        self.configuration = config_class.from_pretrained(self.pretrained_model_path)
        self.tokenizer = tokenizer_class.from_pretrained(self.pretrained_model_path)
        self.model = model_class.from_pretrained(self.pretrained_model_path, config=self.configuration)

        self.model_type = self.configuration.model_type

        self._check_params()

        self.task_prefix = config['task_prefix'] if config['task_prefix'] else ''
        self.task_infix = config['task_infix'] if config['task_infix'] else ''
        self.task_prefix_ids = self.tokenizer.encode(self.task_prefix, add_special_tokens=False)
        self.task_infix_ids = self.tokenizer.encode(self.task_infix, add_special_tokens=False)

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']

        generate_corpus = []

        for src in source_text:
            src_ids = self.tokenizer.encode(src, add_special_tokens=False)

            if self.model_type == 'gpt2':
                input_id = src_ids + self.task_infix_ids
                max_length = self.max_source_length + len(self.task_infix_ids) + self.max_target_length
            else:
                input_id = self.task_prefix_ids + src_ids
                max_length = self.max_target_length

            input_id = torch.tensor(input_id, dtype=torch.long, device=self.device).unsqueeze(1, -1)

            outputs = self.model.generate(input_id, num_beams=5, max_length=max_length, early_stopping=True)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split()

            generate_corpus.append(generated_text)

        return generate_corpus

    def _convert_text_to_tensors(self, source_text, target_text):
        input_ids = []
        labels = []
        attn_masks = []

        for src, tgt in zip(source_text, target_text):
            src_ids = self.tokenizer.encode(src, add_special_tokens=False)
            tgt_ids = self.tokenizer.encode(tgt, add_special_tokens=False)

            if self.model_type == 'gpt2':
                src_ids = src_ids[:self.source_max_length - len(self.task_infix_ids)]
                tgt_ids = tgt_ids[:self.target_max_length - len([self.tokenizer.eos_token_id])]
                input_id = src_ids + self.task_infix_ids + tgt_ids + [self.tokenizer.eos_token_id]
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

        return input_ids, labels, attn_masks

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids, labels, attn_masks = self._convert_text_to_tensors(source_text, target_text)

        outputs = self.model(input_ids, attention_mask=attn_masks, labels=labels)

        return outputs.loss

    def _check_params(self):
        model_type = self.configuration.model_type

        if model_type == 'gpt2':
            self.tokenizer.add_prefix_space = True
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.configuration.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))
