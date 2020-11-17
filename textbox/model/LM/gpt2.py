# @Time   : 2020/11/15
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import UnconditionalGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


class GPT2(UnconditionalGenerator):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(GPT2, self).__init__(config, dataset)

        if config['gpt2_type'] not in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']:
            raise ValueError("The type of GPT2 model should be given in ['gpt2', 'gpt2-medium', 'gpt2-large', "
                             "'gpt2-xl', 'distilgpt2']")
        else:
            self.gpt2_type = config['gpt2_type']

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_type,
                                                       bos_token=dataset.sos_token, eos_token=dataset.eos_token,
                                                       pad_token=dataset.padding_token, unk_token=dataset.eos_token)
        self.configuration = GPT2Config.from_pretrained(self.gpt2_type, output_hidden_states=False)

        self.decoder = GPT2LMHeadModel.from_pretrained(self.gpt2_type, config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.max_length = config['max_seq_length']

    def generate(self, eval_data):
        generate_corpus = []
        number_to_gen = len(eval_data)
        for _ in range(number_to_gen):
            sample_outputs = self.decoder.generate(
                bos_token_id=random.randint(1, 30000),
                do_sample=True,
                top_k=50,
                max_length=self.max_length,
                top_p=0.95,
                num_return_sequences=1
            )
            generated_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
            generate_corpus.append(generated_text.lower().split())
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1):
        input_text = corpus['target_text']
        input_idx = []
        attn_mask = []
        for text in input_text:
            token_list = [self.sos_token] + text + [self.eos_token]
            encodings_dict = self.tokenizer(' '.join(token_list),
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding=True)
            input_idx.append(encodings_dict['input_ids'])
            attn_mask.append(encodings_dict['attention_mask'])
        input_idx = torch.LongTensor(input_idx).to(self.device)
        attn_mask = torch.LongTensor(attn_mask).to(self.device)
        outputs = self.decoder(input_ids=input_idx,
                               labels=input_idx,
                               attention_mask=attn_mask)
        loss = outputs[0]
        return loss
