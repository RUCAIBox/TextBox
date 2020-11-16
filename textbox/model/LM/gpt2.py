# @Time   : 2020/11/15
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import UnconditionalGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


class GPT2(UnconditionalGenerator):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(GPT2, self).__init__(config, dataset)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=dataset.sos_token, eos_token=dataset.eos_token,
                                                       pad_token=dataset.padding_token, unk_token=dataset.eos_token)
        self.configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.max_length = dataset.max_seq_length

    def generate(self, eval_data):
        generate_corpus = []
        number_to_gen = 10
        idx2token = eval_data.idx2token
        for _ in range(number_to_gen):
            generate_token_idx = [self.sos_token_idx]
            generate_tokens = []
            for gen_idx in range(100):
                input_seq = torch.LongTensor([generate_token_idx]).to(self.device)
                input_embedding = self.token_embedder(input_seq) + self.position_embedder(input_seq).to(self.device)
                self_padding_mask = torch.eq(input_seq, self.padding_token_idx).to(self.device)
                self_attn_mask = self.self_attn_mask(input_seq.size(-1)).bool().to(self.device)

                token_logits = self.decoder(input_embedding,
                                            self_padding_mask=self_padding_mask,
                                            self_attn_mask=self_attn_mask)
                token_logits = token_logits[:, -1, :]

                topv, topi = torch.log(F.softmax(token_logits, dim=-1) + 1e-12).data.topk(k=4)
                topi = topi.squeeze()
                token_idx = topi[0].item()
                if token_idx == self.eos_token_idx or gen_idx >= 100:
                    break
                else:
                    generate_token_idx.append(token_idx)
                    generate_tokens.append(idx2token[token_idx])
            generate_corpus.append(generate_tokens)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1):
        input_text = corpus['target_text']
        input_idx = []
        attn_mask = []
        for text in input_text:
            encodings_dict = self.tokenizer(self.sos_token + ' '.join(text) + self.eos_token,
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding='max_length')
            input_idx.append(torch.tensor(encodings_dict['input_ids']))
            attn_mask.append(torch.tensor(encodings_dict['attention_mask']))

        input_idx = torch.tensor(input_idx)
        attn_mask = torch.tensor(attn_mask)
        print(input_idx)
        print(attn_mask)
        exit()
        outputs = self.decoder(input_idx,
                               labels=input_idx,
                               attention_mask=attn_mask,
                               token_type_ids=None)
        loss = outputs[0]
        return loss
