import copy

import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils.rnn import pad_sequence

from textbox.model.abstract_generator import Seq2SeqGenerator

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    BertTokenizer, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel,
    BigBirdTokenizer, BigBirdForCausalLM,
    BertLMHeadModel,
    RobertaTokenizer, RobertaForCausalLM,
    PegasusTokenizer, BigBirdPegasusForConditionalGeneration,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration,
    CpmTokenizer,
    LEDTokenizer, LEDForConditionalGeneration,
    M2M100Tokenizer, M2M100ForConditionalGeneration,
    CTRLTokenizer, CTRLLMHeadModel,
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
    'bert2bert': {
        'tokenizer': BertTokenizer,
        'model': EncoderDecoderModel
    },
    'big_bird_pegasus': {
        'tokenizer': PegasusTokenizer,
        'model': BigBirdPegasusForConditionalGeneration
    },
    'blender_bot': {
        'tokenizer': BlenderbotTokenizer,
        'model': BlenderbotForConditionalGeneration
    },
    'blender_bot_small': {
        'tokenizer': BlenderbotSmallTokenizer,
        'model': BlenderbotSmallForConditionalGeneration
    },
    'led': {
        'tokenizer': LEDTokenizer,
        'model': LEDForConditionalGeneration
    },
    'm2m100': {
        'tokenizer': M2M100Tokenizer,
        'model': M2M100ForConditionalGeneration
    },

    'gpt2': {
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel
    },
    'big_bird': {
        'tokenizer': BigBirdTokenizer,
        'model': BigBirdForCausalLM
    },
    'bert': {
        'tokenizer': BertTokenizer,
        'model': BertLMHeadModel
    },
    'roberta': {
        'tokenizer': RobertaTokenizer,
        'model': RobertaForCausalLM
    },
    'cpm': {
        'tokenizer': CpmTokenizer,
        'model': GPT2LMHeadModel
    },
    'ctrl': {
        'tokenizer': CTRLTokenizer,
        'model': CTRLLMHeadModel
    },
    'dialo_gpt': {
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel
    },
}

CLM_MODELS = ['gpt2', 'big_bird', 'bert', 'roberta', 'cpm', 'ctrl', 'dialo_gpt']

EncDecLM_MODELS = ['t5', 'bart', 'bert2bert', 'big_bird_pegasus', 'blender_bot', 'blender_bot_small', 'led', 'm2m100']


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
            init_kwargs = {'is_decoder': True} if self.model_name in CLM_MODELS else {}
            self.model = model_class.from_pretrained(self.model_name_or_path, **init_kwargs)

        self.configuration = self.model.config
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path)

        self._init_params()

        self.prefix = config['prefix_prompt'] if config['prefix_prompt'] else ''
        self.suffix = config['suffix_prompt'] if config['suffix_prompt'] else ''

        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        if self.model_name in CLM_MODELS:
            # set bos_id & eos_id for CLM
            if self.tokenizer.cls_token:  # big_bird, bert, roberta, cpm
                self.bos_token_id = [self.tokenizer.cls_token_id]
            elif self.tokenizer.bos_token:  # gpt2, dialo_gpt
                self.bos_token_id = [self.tokenizer.bos_token_id]
            else:  # ctrl
                self.bos_token_id = []

            if self.tokenizer.sep_token:  # big_bird, bert, roberta, cpm
                self.eos_token_id = [self.tokenizer.sep_token_id]
            elif self.tokenizer.eos_token:  # gpt2, ctrl, dialo_gpt
                self.eos_token_id = [self.tokenizer.eos_token_id]
            else:
                raise ValueError("eos token id is not set yet, check _init_params() first")

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

            if self.model_name in CLM_MODELS:
                input_id, label = self._casual_model_encode(src_ids, tgt_ids)
            else:
                input_id, label = self._encoder_decoder_model_encode(src_ids, tgt_ids)

            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            attn_masks.append(torch.ones(len(input_id), dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0).to(self.device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(self.device)

        inputs = {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}
        return inputs

    def forward(self, corpus, epoch_idx=-1):
        inputs = self._generate_default_inputs(corpus)
        outputs = self.model(**inputs)
        return outputs.loss

    def _init_params(self):
        if isinstance(self.tokenizer, GPT2Tokenizer):  # gpt2, roberta, bart, led
            self.tokenizer.add_prefix_space = True

        if self.model_name == 'gpt2' or self.model_name == 'dialo_gpt':
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.model_name == 'ctrl':
            self.tokenizer.add_special_tokens(({'eos_token': '</s>'}))
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.build_inputs_with_special_tokens = lambda t0, t1: t0 + [self.tokenizer.eos_token_id]
        elif self.model_name == 'bart' or self.model_name == 'led':
            self.configuration.forced_bos_token_id = self.tokenizer.bos_token_id
        elif self.model_name == 'bert2bert':
            self.configuration.decoder_start_token_id = self.tokenizer.cls_token_id
            self.configuration.pad_token_id = self.tokenizer.pad_token_id
        elif self.model_name == 'm2m100':
            self.configuration.forced_bos_token_id = self.tokenizer.get_lang_id(self.config['tgt_lang'])
            self.tokenizer.src_lang = self.config['src_lang']
            self.tokenizer.tgt_lang = self.config['tgt_lang']
        elif self.model_name == 'blender_bot_small':  # num_special_tokens_to_add() == 0
            self.tokenizer.build_inputs_with_special_tokens = lambda t0, t1: t0 + [self.tokenizer.eos_token_id]

    def _encoder_decoder_model_encode(self, src_ids, tgt_ids):
        """
        t5: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        bart, led: [<s>, src, </s>], [<s>, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: <s>
        bert2bert: [[CLS], src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [CLS]
        big_bird_pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        blender_bot: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        blender_bot_small: [src, __end__], [tgt, __end__], decoder_start_token_id: __start__
        m2m100: [src_lang_id, src, </s>], [tgt_lang_id, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: tgt_lang_id
        """
        assert self.configuration.pad_token_id
        assert self.configuration.decoder_start_token_id

        src_ids = src_ids[:self.source_max_length - self.tokenizer.num_special_tokens_to_add()
                          - len(self.prefix_ids) - len(self.suffix_ids)]
        tgt_ids = tgt_ids[:self.target_max_length - self.tokenizer.num_special_tokens_to_add()]
        input_id = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + src_ids + self.suffix_ids)
        if self.model_name == 'm2m100':
            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)
        else:
            label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)

        if self.model_name == 'bert2bert':
            label = label[1:]

        return input_id, label

    def _casual_model_encode(self, src_ids, tgt_ids):
        """
        gpt2, dialo_gpt: [<|endoftext|>, src, <|endoftext|>, tgt, <|endoftext|>]
        big_bird, bert: [[CLS], src, [SEP], tgt, [SEP]]
        roberta: [<s>, src, </s>, tgt, </s>]
        cpm: [<cls>, src, <sep>, tgt, <sep>]
        ctrl: [src, </s>, tgt, </s>]
        """
        src_ids = src_ids[:self.source_max_length-len(self.prefix_ids)-len(self.suffix_ids)-1-len(self.bos_token_id)]
        tgt_ids = tgt_ids[:self.target_max_length - 1]
        src_input_id = self.bos_token_id + self.prefix_ids + src_ids + self.suffix_ids + self.eos_token_id
        tgt_input_id = tgt_ids + self.eos_token_id
        input_id = src_input_id + tgt_input_id
        label = len(src_input_id) * [-100] + tgt_input_id

        return input_id, label
