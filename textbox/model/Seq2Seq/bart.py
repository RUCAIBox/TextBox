# @Time   : 2020/11/16
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import ConditionalGenerator
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration


class BART(ConditionalGenerator):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(BART, self).__init__(config, dataset)

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',
                                                       bos_token=dataset.sos_token,
                                                       eos_token=dataset.eos_token,
                                                       pad_token=dataset.padding_token,
                                                       unk_token=dataset.unknown_token,
                                                       add_prefix_space=True)
        self.configuration = BartConfig.from_pretrained('facebook/bart-base')

        self.decoder = BartForConditionalGeneration.from_pretrained("gpt2", config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length']

    def generate(self, eval_dataloader):
        generate_corpus = []
        for batch_data in eval_dataloader:
            source_text_list = [' '.join(token_list) for token_list in batch_data['source_text']]
            sample_outputs = self.decoder.generate(
                source_text_list,
                num_beams=4,
                max_length=self.max_target_length,
                early_stopping=True
            )
            generated_text = [self.tokenizer.decode(sample, skip_special_tokens=True) for sample in sample_outputs]
            generated_text = [text.lower().split() for text in generated_text]
            generate_corpus.extend(generated_text)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        source_text_list = []
        target_text_list = []
        for tid in range(len(source_text)):
            source_text_list.append(' '.join(source_text[tid]))
            target_token_list = [self.sos_token] + target_text[tid] + [self.eos_token]
            target_text_list.append(' '.join(target_token_list))
        encoding_dict = self.tokenizer(src_texts=source_text_list,
                                       tgt_texts=target_text_list,
                                       max_length=self.max_source_length,
                                       max_target_length=self.max_target_length,
                                       padding=True)
        input_idx = torch.LongTensor(encoding_dict['input_ids']).to(self.device)
        attn_mask = torch.LongTensor(encoding_dict['attention_mask']).to(self.device)
        decoder_input_idx = torch.LongTensor(encoding_dict['labels']).to(self.device)
        outputs = self.decoder(input_idx,
                               attention_mask=attn_mask,
                               decoder_input_ids=decoder_input_idx)
        loss = outputs[0]
        return loss
