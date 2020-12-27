# @Time   : 2020/12/25
# @Author : Puzhao Xie
# @Email  : xiepuzhao@ruc.edu.cn


import torch
import torch.nn as nn
from textbox.model.abstract_generator import ConditionalGenerator
from transformers import BertTokenizer, EncoderDecoderConfig, BertConfig, BertGenerationEncoder, BertGenerationDecoder, \
    EncoderDecoderModel

'''
Reference:  Leveraging Pre-trained Checkpoints for Sequence Generation Tasks 
Code Reference: https://huggingface.co/transformers/model_doc/bertgeneration.html
'''


class BERT2BERT(ConditionalGenerator):
    r"""The BertGeneration model is a BERT model that can be leveraged for sequence-to-sequence tasks using
    EncoderDecoderModel as proposed in Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by
    Sascha Rothe, Shashi Narayan, Aliaksei Severyn
    """

    def __init__(self, config, dataset):
        super(BERT2BERT, self).__init__(config, dataset)

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased')
        self.encoder_configure = BertConfig.from_pretrained(
            'bert-base-cased')

        self.decoder_configure = BertConfig.from_pretrained(
            'bert-base-cased')

        self.encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=self.encoder_configure, decoder_config=self.decoder_configure)

        self.encoder = BertGenerationEncoder.from_pretrained(
            'bert-base-cased',
            bos_token_id=101, eos_token_id=102)

        self.decoder = BertGenerationDecoder.from_pretrained(
            'bert-base-cased',
            add_cross_attention=True,
            is_decoder=True,
            bos_token_id=101,
            eos_token_id=102)

        self.encoder_decoder = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder,
                                                   config=self.encoder_decoder_config)

        self.sos_token = dataset.sos_token
        self.eos_token = dataset.eos_token
        self.padding_token_idx = self.tokenizer.pad_token_id
        self.max_source_length = config['source_max_seq_length']
        self.max_target_length = config['target_max_seq_length']

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, eval_dataloader):
        generate_corpus = []
        with torch.no_grad():
            for batch_data in eval_dataloader:
                source_text = batch_data["source_text"]
                for text in source_text:
                    sentence = ' '.join(text)
                    encoding_dict = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
                    input_ids = encoding_dict['input_ids'].to(self.device)
                    sample_outputs = self.encoder_decoder.generate(input_ids,
                                                                   num_beams=4,
                                                                   max_length=self.max_target_length,
                                                                   early_stopping=True,
                                                                   bos_token_id=101,
                                                                   eos_token_id=102)
                    generated_text = [self.tokenizer.decode(sample, skip_special_tokens=True) for sample in
                                      sample_outputs]
                    generated_text = [text.lower().split() for text in generated_text]
                    generate_corpus.extend(generated_text)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids = []
        attn_masks = []
        for text in source_text:
            sentence = ' '.join(text)
            encoding_dict = self.tokenizer(sentence,
                                           max_length=self.max_source_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt",
                                           add_special_tokens=False)
            input_ids.append(encoding_dict['input_ids'])
            attn_masks.append(encoding_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attn_masks = torch.cat(attn_masks, dim=0).to(self.device)
        target_ids = []
        for text in target_text:
            sentence = ' '.join(text)
            encoding_dict = self.tokenizer(sentence,
                                           max_length=self.max_target_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt")
            target_ids.append(encoding_dict['input_ids'])
        target_ids = torch.cat(target_ids, dim=0).to(self.device)
        target_ids = target_ids.contiguous()
        outputs = self.encoder_decoder(input_ids,
                                       attention_mask=attn_masks,
                                       decoder_input_ids=target_ids,
                                       labels=target_ids)

        token_logits = outputs[1]
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_ids.view(-1))
        loss = loss.reshape_as(target_ids)

        length = (target_ids != self.padding_token_idx).sum(dim=1).float()
        loss = loss.sum(dim=1) / length

        return loss.mean()
