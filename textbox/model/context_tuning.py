import torch
import torch.nn as nn
import torch.nn.functional as F
from .pretrained_models import Pretrained_Models
from transformers import BertTokenizer, BertForMaskedLM, BertModel, BartModel, RobertaTokenizer, RobertaForMaskedLM


class Context_Tuning(Pretrained_Models):

    def __init__(self, config, tokenizer):
        super(Context_Tuning, self).__init__(config, tokenizer)
        self.prefix_prompt = config['prefix_prompt'] or ''
        self.suffix_prompt = config['suffix_prompt'] or ''

        self.prompt_generator = config['prompt_generator']
        if self.prompt_generator == 'roberta':
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model_path'])
            self.roberta_model = RobertaForMaskedLM.from_pretrained(config['roberta_model_path'])
        elif self.prompt_generator == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_model_path'])
            self.semantic_mapping = config['semantic_mapping']
            if self.semantic_mapping:
                self.bert_model = BertForMaskedLM.from_pretrained(config['bert_model_path'])
            else:
                self.bert_model = BertModel.from_pretrained(config['bert_model_path'])
        elif self.prompt_generator == 'bart':
            self.bart_model = BartModel.from_pretrained(config['model_path']).get_encoder()

        if config['bitfit']:
            for para in self.parameters():
                if len(para.shape) > 1:
                    para.requires_grad_(False)

    def truncate_input(self, texts, tokenizer):
        return tokenizer.batch_decode(
            tokenizer(texts, max_length=self.config['src_len'], truncation=True, add_special_tokens=False)['input_ids']
        )

    def _process_prompt_tuning_input(self, inputs, batch):
        batch_size = len(batch['source_text'])

        if self.prompt_generator == 'bert':
            masks = self.bert_tokenizer.mask_token * self.prompt_length
            tmp_source_text = self.truncate_input(batch['source_text'], self.bert_tokenizer)
            texts = [masks + self.prefix_prompt + t + self.suffix_prompt + masks for t in tmp_source_text]
            bert_inputs = self.bert_tokenizer(
                texts,
                max_length=self.bert_tokenizer.model_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = bert_inputs['input_ids'].to(self.device)
            attn_masks = bert_inputs['attention_mask'].to(self.device)
            outputs = self.bert_model(input_ids, attn_masks)
            if self.semantic_mapping:
                outputs = F.softmax(outputs.logits, dim=-1)  # b, l, v
                hidden_states = outputs.matmul(self.bert_model.bert.get_input_embeddings().weight)  # b, l, e
            else:
                hidden_states = outputs.last_hidden_state
            prompt_embeds = hidden_states[input_ids == self.bert_tokenizer.mask_token_id]  # b*2*pl, e
            prompt_embeds = prompt_embeds.reshape(batch_size, 2, self.prompt_length, -1)  # b, 2, pl, e
        elif self.prompt_generator == 'roberta':
            masks = self.roberta_tokenizer.mask_token * self.prompt_length
            tmp_source_text = self.truncate_input(batch['source_text'], self.roberta_tokenizer)
            texts = [masks + self.prefix_prompt + t + self.suffix_prompt + masks for t in tmp_source_text]
            roberta_inputs = self.roberta_tokenizer(
                texts,
                max_length=self.roberta_tokenizer.model_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = roberta_inputs['input_ids'].to(self.device)
            attn_masks = roberta_inputs['attention_mask'].to(self.device)
            outputs = self.roberta_model(input_ids, attn_masks)
            outputs = F.softmax(outputs.logits, dim=-1)  # b, l, v
            hidden_states = outputs.matmul(self.roberta_model.roberta.get_input_embeddings().weight)  # b, l, e
            prompt_embeds = hidden_states[input_ids == self.roberta_tokenizer.mask_token_id]  # b*2*pl, e
            prompt_embeds = prompt_embeds.reshape(batch_size, 2, self.prompt_length, -1)  # b, 2, pl, e
        elif self.prompt_generator == 'bart':
            masks = self.tokenizer.mask_token * self.prompt_length
            tmp_source_text = self.truncate_input(batch['source_text'], self.tokenizer)
            texts = [masks + self.prefix_prompt + t + self.suffix_prompt + masks for t in tmp_source_text]
            bart_inputs = self.tokenizer(
                texts, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
            )
            input_ids = bart_inputs['input_ids'].to(self.device)
            attn_masks = bart_inputs['attention_mask'].to(self.device)
            outputs = self.bart_model(input_ids, attn_masks)
            hidden_states = outputs.last_hidden_state  # b, l, e
            prompt_embeds = hidden_states[input_ids == self.tokenizer.mask_token_id]  # b*2*pl, e
            prompt_embeds = prompt_embeds.reshape(batch_size, 2, self.prompt_length, -1)  # b, 2, pl, e

        inputs_embeds = self.model.get_input_embeddings()(inputs['input_ids'])  # b, l, e
        inputs_embeds = torch.cat([prompt_embeds[:, 0], inputs_embeds, prompt_embeds[:, 1]], dim=1)  # b, pl+l+pl, e
        inputs['inputs_embeds'] = inputs_embeds
        del inputs['input_ids']
        mask = torch.ones(batch_size, self.prompt_length, dtype=torch.long).to(self.device)
        inputs['attention_mask'] = torch.cat([mask, inputs['attention_mask'], mask], dim=1)
        return inputs
