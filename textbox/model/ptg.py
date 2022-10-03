import torch
import torch.nn as nn
import torch.nn.functional as F
from .pretrained_models import Pretrained_Models
from transformers import AutoTokenizer, AutoModel

source_task_set = {
    'cross-task1': ['squad', 'wiki', 'quora', 'wp', 'cnndm'],
    'cross-task2': ['squad', 'wiki', 'quora', 'wp', 'pc'],
    'cross-dataset1': ['msn', 'mn', 'nr'],
    'cross-dataset2': ['tc', 'da', 'mw'],
}


class PTG(Pretrained_Models):

    def __init__(self, config, tokenizer):
        super(PTG, self).__init__(config, tokenizer)

        prompt_source = torch.load(config['prompt_source_path'])
        source_task = config['source_task']
        self.lam = config['ptg_lambda']
        if source_task in source_task_set:
            source_task = source_task_set[source_task]
        self.task_embedding = [prompt_source[task] for task in source_task]
        self.task_embedding = torch.stack(self.task_embedding).to(self.device)  # tn, pl, e
        self.embedding_size = self.model.config.hidden_size

        assert self.model.config.hidden_size == self.task_embedding.size(-1), "PTG only supports BART-large!"

        self.bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model_path'])
        self.bert_model = AutoModel.from_pretrained(config['bert_model_path'])

        self.task_num = self.task_embedding.size(0)
        self.head_num = self.model.config.num_attention_heads
        self.head_dim = self.embedding_size // self.head_num
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_proj = nn.Linear(self.embedding_size, self.embedding_size)
        self.q_proj = nn.Linear(self.embedding_size, self.embedding_size)
        self.out_proj = nn.Linear(self.embedding_size, self.embedding_size)
        self.task_key = nn.Embedding(self.task_num + 1, self.embedding_size)  # tn+1, e
        self.model.requires_grad_(True)
        self.bert_model.requires_grad_(False)

    def sentence_embedding(self, text):
        encoding_dict = self.bert_tokenizer(
            text, max_length=self.bert_tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = encoding_dict['input_ids'].to(self.device)
        attn_masks = encoding_dict['attention_mask'].to(self.device)
        output = self.bert_model(input_ids, attn_masks)['last_hidden_state']  # b, l, h
        hidden_state = output * attn_masks.unsqueeze(-1)
        embedding = hidden_state.sum(dim=1) / attn_masks.sum(dim=1).unsqueeze(-1)
        return embedding.detach()

    def MHA(self, query, key, value):
        batch_size = key.size(0)
        # b*h, 1, d
        query = query.reshape(batch_size, -1, self.head_num,
                              self.head_dim).transpose(1, 2).reshape(batch_size * self.head_num, -1, self.head_dim)
        # b*h, tn, d
        key = key.reshape(batch_size, -1, self.head_num,
                          self.head_dim).transpose(1, 2).reshape(batch_size * self.head_num, -1, self.head_dim)
        # b*h, tn, pl*d
        value = value.reshape(batch_size, self.task_num, -1, self.head_num,
                              self.head_dim).permute(0, 3, 1, 2,
                                                     4).reshape(batch_size * self.head_num, self.task_num, -1)

        attn_weights = torch.bmm(query, key.transpose(1, 2)) * self.scaling  # b*h, 1, tn
        attn_probs = F.dropout(attn_weights, p=0.1)
        attn_probs = F.softmax(attn_probs, dim=-1)  # b*h, 1, tn
        attn_output = torch.bmm(attn_probs, value)  # b*h, 1, pl*d
        prompt_embedding = attn_output.squeeze(1).reshape(batch_size, self.head_num, -1, self.head_dim)  # b, h, pl, d
        prompt_embedding = prompt_embedding.transpose(1, 2).reshape(batch_size, -1, self.embedding_size)  # b, pl, e
        prompt_embedding = self.out_proj(prompt_embedding)  # b, pl, e
        return prompt_embedding

    def _process_prompt_tuning_input(self, inputs, batch):
        input_ids = inputs['input_ids']
        batch_size = input_ids.size(0)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)  # b, l, e

        task_key = self.task_key.weight.repeat(batch_size, 1, 1)  # b, tn+1, e
        task_query = self.q_proj(task_key[:, -1:])  # b, 1, e
        key = self.k_proj(task_key[:, :-1])  # b, tn, e
        value = self.v_proj(self.task_embedding).reshape(self.task_num, -1).repeat(batch_size, 1, 1)  # b, tn, pl*e
        input_query = self.sentence_embedding(batch['source_text']).unsqueeze(1)  # b, 1, e
        prompt_embeds = self.lam * self.MHA(task_query, key, value) + (1 - self.lam) * self.MHA(input_query, key, value)

        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        inputs['inputs_embeds'] = inputs_embeds
        del inputs['input_ids']
        mask = torch.ones(batch_size, self.prompt_length, dtype=torch.long).to(self.device)
        inputs['attention_mask'] = torch.cat([mask, inputs['attention_mask']], dim=1)
        return inputs
