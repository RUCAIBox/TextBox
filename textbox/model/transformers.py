r"""
################################################
    Integration of Encoder-Decoder LM and Casual LM in transformers API

    Summary of Related Papers:
    gpt2: Language Models are Unsupervised Multitask Learners
    gpt: Improving Language Understanding by Generative Pre-Training
    big_bird, big_bird_pegasus: Big Bird: Transformers for Longer Sequences
    bert: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    roberta: RoBERTa: A Robustly Optimized BERT Pretraining Approach
    cpm: CPM: A Large-scale Generative Chinese Pre-trained Language Model
    ctrl: CTRL: A Conditional Transformer Language Model for Controllable Generation
    megatron_bert: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
    transfo_xl: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
    gpt_neo: The Pile: An 800GB Dataset of Diverse Text for Language Modeling

    t5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    mt5: mT5: A massively multilingual pre-trained text-to-text transformer
    bart: BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
    led: Longformer: The Long-Document Transformer
    mbart: Multilingual Denoising Pre-training for Neural Machine Translation
    bert2bert: Leveraging Pre-trained Checkpoints for Sequence Generation Tasks
    pegasus: PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
    blender_bot, blender_bot_small: Recipes for building an open-domain chatbot
    m2m100: Beyond English-Centric Multilingual Machine Translation
    prophet_net: ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training
"""


import torch
import warnings
import torch.nn.functional as F
from torch import Tensor
from typing import List
from torch.nn.utils.rnn import pad_sequence


from textbox.model.abstract_model import AbstractModel

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    BertTokenizer, EncoderDecoderModel,
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
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel,
    MegatronBertForCausalLM,
    TransfoXLTokenizer, TransfoXLLMHeadModel,
    MBartTokenizer, MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    PegasusForConditionalGeneration,
    ProphetNetTokenizer, ProphetNetForConditionalGeneration,
    GPTNeoForCausalLM
)

MODEL_CLASSES = {
    # EncDecLM_MODELS
    't5': {
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration
    },
    'mt5': {
        'tokenizer': T5Tokenizer,
        'model': MT5ForConditionalGeneration
    },
    'bart': {
        'tokenizer': BartTokenizer,
        'model': BartForConditionalGeneration
    },
    'led': {
        'tokenizer': LEDTokenizer,
        'model': LEDForConditionalGeneration
    },
    'mbart': {
        'tokenizer': MBartTokenizer,
        'model': MBartForConditionalGeneration
    },
    'bert2bert': {
        'tokenizer': BertTokenizer,
        'model': EncoderDecoderModel
    },
    'big_bird_pegasus': {
        'tokenizer': PegasusTokenizer,
        'model': BigBirdPegasusForConditionalGeneration
    },
    'pegasus': {
        'tokenizer': PegasusTokenizer,
        'model': PegasusForConditionalGeneration
    },
    'blender_bot': {
        'tokenizer': BlenderbotTokenizer,
        'model': BlenderbotForConditionalGeneration
    },
    'blender_bot_small': {
        'tokenizer': BlenderbotSmallTokenizer,
        'model': BlenderbotSmallForConditionalGeneration
    },
    'm2m100': {
        'tokenizer': M2M100Tokenizer,
        'model': M2M100ForConditionalGeneration
    },
    'prophet_net': {
        'tokenizer': ProphetNetTokenizer,
        'model': ProphetNetForConditionalGeneration
    },

    # CLM_MODELS
    'gpt2': {
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel
    },
    'gpt': {
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel
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
    'megatron_bert': {
        'tokenizer': BertTokenizer,
        'model': MegatronBertForCausalLM
    },
    'transfo_xl': {
        'tokenizer': TransfoXLTokenizer,
        'model': TransfoXLLMHeadModel
    },
    'gpt_neo': {
        'tokenizer': GPT2Tokenizer,
        'model': GPTNeoForCausalLM
    }
}

CLM_MODELS = ['gpt2', 'gpt', 'big_bird', 'bert', 'roberta', 'cpm', 'ctrl', 'megatron_bert', 'transfo_xl', 'gpt_neo']

EncDecLM_MODELS = ['t5', 'mt5', 'bart', 'led', 'mbart', 'bert2bert', 'big_bird_pegasus', 'pegasus', 'blender_bot',
                   'blender_bot_small', 'm2m100', 'prophet_net']


def _pad_sequence(tensors: List[Tensor], padding_value: int, padding_side: str = 'right'):
    """
    Pad encoded inputs (on left/right and up to max length in the batch)
    """
    max_len = max(tensor.size()[0] for tensor in tensors)
    padded_tensors = []
    if padding_side == 'right':
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    elif padding_side == 'left':
        for tensor in tensors:
            padding_length = max_len - len(tensor)
            padded_tensor = torch.cat([torch.full([padding_length], padding_value, dtype=tensor.dtype), tensor], dim=-1)
            padded_tensors.append(padded_tensor)
    else:
        raise ValueError("Invalid padding strategy:" + str(padding_side))
    padded_tensors = torch.stack(padded_tensors, dim=0)
    return padded_tensors


class Transformers(AbstractModel):
    def __init__(self, config, dataset):
        super(Transformers, self).__init__(config, dataset)
        self.config = config

        self.model_name_or_path = config['pretrained_model_path']
        self.model_name = config['model'].lower()
        self.is_casual_model = bool(self.model_name in CLM_MODELS)
        self.is_enc_dec_model = bool(self.model_name in EncDecLM_MODELS)
        assert self.is_casual_model or self.is_enc_dec_model, "model must be either CLMs or EncDecLMs"

        tokenizer_class = MODEL_CLASSES[self.model_name]['tokenizer']
        model_class = MODEL_CLASSES[self.model_name]['model']

        if self.model_name == 'bert2bert':
            self.model = model_class.from_encoder_decoder_pretrained(self.model_name_or_path, self.model_name_or_path)
        else:
            init_kwargs = {'is_decoder': True} if self.is_casual_model else {}
            self.model = model_class.from_pretrained(self.model_name_or_path, **init_kwargs)

        self.configuration = self.model.config
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path)

        self._process_prompt()
        self._init_params()
        if self.is_casual_model:
            self._prepare_bos_eos_token_for_casual_model()

        self.label_smoothing = config['label_smoothing'] if config['label_smoothing'] else 0.
        self.truncate = config['truncate'] or 'tail'
        self.num_beams = config['num_beams'] or 4
        self.src_ids_num = \
            self.source_max_length-len(self.prefix_ids)-len(self.suffix_ids)-1-len(self.bos_token_id) \
            if self.is_casual_model else \
            self.source_max_length-self.tokenizer.num_special_tokens_to_add()-len(self.prefix_ids)-len(self.suffix_ids)
        self.tgt_ids_num = self.target_max_length - 1 \
            if self.is_casual_model else self.target_max_length - self.tokenizer.num_special_tokens_to_add()

    def _process_prompt(self):
        r"""
        Prompts can be added to the beginning and end of **source ids**.
        """
        self.prefix = self.config['prefix_prompt'] if self.config['prefix_prompt'] else ''
        self.suffix = self.config['suffix_prompt'] if self.config['suffix_prompt'] else ''

        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _init_params(self):
        if isinstance(self.tokenizer, GPT2Tokenizer):  # gpt2, roberta, bart, led
            self.tokenizer.add_prefix_space = True

        # (1): tokenizer needs to add eos token
        if self.model_name in ['ctrl', 'gpt']:
            self.tokenizer.add_special_tokens(({'eos_token': '</s>'}))
            self.model.resize_token_embeddings(len(self.tokenizer))

        # (2): tokenizer needs to add pad token
        if self.model_name in ['gpt2', 'transfo_xl', 'gpt_neo', 'ctrl']:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # (3): tokenizer needs to modify build_inputs_with_special_tokens()
        if self.model_name in ['gpt2', 'transfo_xl', 'blender_bot_small', 'gpt_neo']:
            self.tokenizer.build_inputs_with_special_tokens = lambda t0, t1=None: t0 + [self.tokenizer.eos_token_id]

        # (4): tokenizer needs to set src_lang, tgt_lang (used in translation task)
        if self.model_name in ['m2m100', 'mbart']:
            assert self.config['src_lang'] and self.config['tgt_lang'], \
                self.model_name + ' needs to specify source language and target language'
            self.tokenizer.src_lang = self.config['src_lang']
            self.tokenizer.tgt_lang = self.config['tgt_lang']

        # (5): model specific init
        if self.model_name in ['bart', 'led']:
            self.configuration.forced_bos_token_id = self.tokenizer.bos_token_id
        elif self.model_name == 'bert2bert':
            self.configuration.decoder_start_token_id = self.tokenizer.cls_token_id
            self.configuration.pad_token_id = self.tokenizer.pad_token_id
        elif self.model_name == 'm2m100':
            self.configuration.forced_bos_token_id = self.tokenizer.get_lang_id(self.config['tgt_lang'])

    def _prepare_bos_eos_token_for_casual_model(self):
        """
        BOS = cls_token: big_bird, bert, roberta, cpm, megatron_bert
            = bos_token: gpt2, gpt_neo
            = None: ctrl, gpt, transfo_xl

        EOS = sep_token: big_bird, bert, roberta, cpm, megatron_bert
            = eos_token: gpt2, ctrl, transfo_xl, gpt, gpt_neo
        """

        if self.tokenizer.cls_token:
            self.bos_token_id = [self.tokenizer.cls_token_id]
        elif self.tokenizer.bos_token:
            self.bos_token_id = [self.tokenizer.bos_token_id]
        else:
            self.bos_token_id = []

        if self.tokenizer.sep_token:
            self.eos_token_id = [self.tokenizer.sep_token_id]
        elif self.tokenizer.eos_token:
            self.eos_token_id = [self.tokenizer.eos_token_id]
        else:
            raise ValueError("eos token id is not set yet, check _init_params() first")

        self.configuration.eos_token_id = self.eos_token_id[0]  # used in generate() for casual models

    def _generate_default_inputs(self, corpus):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        input_ids = []
        labels = []
        attn_masks = []

        pad_token_id = self.tokenizer.pad_token_id
        padding_side = self.tokenizer.padding_side

        for src, tgt in zip(source_text, target_text):
            if self.is_casual_model:
                input_id, label = self._casual_model_encode(src, tgt_text=tgt)
            else:
                input_id, label = self._encoder_decoder_model_encode(src, tgt_text=tgt)

            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            attn_masks.append(torch.ones(len(input_id), dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))

        input_ids = _pad_sequence(input_ids, padding_value=pad_token_id, padding_side=padding_side).to(self.device)
        attn_masks = _pad_sequence(attn_masks, padding_value=0, padding_side=padding_side).to(self.device)
        labels = _pad_sequence(labels, padding_value=-100, padding_side=padding_side).to(self.device)

        inputs = {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}

        processed_inputs = self._inputs_postprocess(inputs)
        return processed_inputs

    def _encoder_decoder_model_encode(self, src_text, tgt_text=None, eval=False):
        r"""
        t5, mt5: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        bart, led: [<s>, src, </s>], [<s>, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: <s>
        bert2bert: [[CLS], src, [SEP]], [[CLS](**remove later**), tgt, [SEP]], decoder_start_token_id: [CLS]
        big_bird_pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        blender_bot: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        blender_bot_small: [src, __end__], [tgt, __end__], decoder_start_token_id: __start__
        m2m100: [src_lang_id, src, </s>], [tgt_lang_id, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: tgt_lang_id
        mbart: [src, </s>, src_lang_id], [tgt, </s>, tgt_lang_id], decoder_start_token_id: tgt_lang_id
        prophet_net: [src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [SEP]
        """

        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        src_ids = src_ids[:self.src_ids_num] if self.truncate == 'tail' else src_ids[-self.src_ids_num:]
        input_id = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + src_ids + self.suffix_ids)
        if eval:
            return input_id

        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=False)
        tgt_ids = tgt_ids[:self.tgt_ids_num]
        if self.model_name in ['m2m100', 'mbart']:
            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)
        else:
            label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)
        return input_id, label

    def _casual_model_encode(self, src_text, tgt_text=None, eval=False):
        r"""
        gpt2, gpt_neo: [<|endoftext|>, src, <|endoftext|>, tgt, <|endoftext|>]
        big_bird, bert, megatron_bert: [[CLS], src, [SEP], tgt, [SEP]]
        roberta: [<s>, src, </s>, tgt, </s>]
        cpm: [src, <sep>, tgt, <sep>, <cls>]
        ctrl, gpt: [src, </s>, tgt, </s>]
        transfo_xl: [src, <eos>, tgt, <eos>]
        """

        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        src_ids = src_ids[:self.src_ids_num] if self.truncate == 'tail' else src_ids[-self.src_ids_num:]
        src_ids = self.prefix_ids + src_ids + self.suffix_ids

        if self.tokenizer.padding_side == 'left':  # cpm
            src_input_id = src_ids + self.eos_token_id
        else:
            src_input_id = self.bos_token_id + src_ids + self.eos_token_id
        if eval:
            return src_input_id

        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=False)
        tgt_ids = tgt_ids[:self.tgt_ids_num]
        if self.tokenizer.padding_side == 'left':  # cpm
            tgt_input_id = tgt_ids + self.eos_token_id + self.bos_token_id
        else:
            tgt_input_id = tgt_ids + self.eos_token_id

        input_id = src_input_id + tgt_input_id
        label = len(src_input_id) * [-100] + tgt_input_id

        return input_id, label

    def _inputs_postprocess(self, inputs):
        # model specific process
        if self.model_name == 'transfo_xl':
            inputs.pop('attention_mask')  # transfo_xl construct mask inside the model
        elif self.model_name == 'bert2bert':
            inputs['labels'] = inputs['labels'][:, 1:]  # remove the decoder_start_token_id: [CLS]

        return inputs

    def _compute_loss(self, outputs, labels, ignore_index=-100):
        if self.model_name == 'transfo_xl':
            loss = outputs.losses.mean()
            if self.label_smoothing > 0:
                warnings.warn("label smoothing for transformer-xl is not implemented")  # no logits returned
            return loss
        else:
            loss = outputs.loss

        if self.label_smoothing > 0:
            if self.is_casual_model:
                logits = outputs.logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous().view(-1)
            else:
                logits = outputs.logits
                labels = labels.view(-1)

            probs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))  # b * l, v
            probs = -probs.mean(dim=-1)  # b * l
            smooth_loss = probs[labels != ignore_index].mean()
            loss = loss * (1 - self.label_smoothing) + smooth_loss * self.label_smoothing

        return loss

    def forward(self, corpus, epoch_idx=-1):
        inputs = self._generate_default_inputs(corpus)
        outputs = self.model(**inputs)
        loss = self._compute_loss(outputs, inputs['labels'])
        return loss

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']

        pad_token_id = self.tokenizer.pad_token_id
        padding_side = self.tokenizer.padding_side
        decode_params = {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}
        generate_params = {'num_beams': self.num_beams, 'max_length': self.target_max_length,
                           'do_sample': True, 'early_stopping': True}

        if self.is_casual_model:
            input_ids = []
            attn_masks = []
            for src in source_text:
                input_id = self._casual_model_encode(src, eval=True)
                input_ids.append(torch.tensor(input_id, dtype=torch.long))
                attn_masks.append(torch.ones(len(input_id), dtype=torch.long))

            input_ids = _pad_sequence(input_ids, padding_value=pad_token_id, padding_side='left').to(self.device)
            attn_masks = _pad_sequence(attn_masks, padding_value=0, padding_side='left').to(self.device)
            input_id_len = input_ids.shape[1]
            generate_params['max_length'] += input_id_len
            sample_outputs = self.model.generate(input_ids, attention_mask=attn_masks, **generate_params)
            generated_text = self.tokenizer.batch_decode(sample_outputs[:, input_id_len:], **decode_params)
            generate_corpus = [text.split() for text in generated_text]
            return generate_corpus

        else:
            input_ids = [torch.tensor(self._encoder_decoder_model_encode(src, eval=True), dtype=torch.long)
                         for src in source_text]
            attn_masks = [torch.ones(len(inp), dtype=torch.long) for inp in input_ids]
            input_ids = _pad_sequence(input_ids, padding_value=pad_token_id, padding_side=padding_side).to(self.device)
            attn_masks = _pad_sequence(attn_masks, padding_value=0, padding_side=padding_side).to(self.device)

            if self.model_name == 'mbart':
                decoder_start_token_id = self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
            else:
                decoder_start_token_id = self.configuration.decoder_start_token_id

            sample_outputs = self.model.generate(
                input_ids,
                attention_mask=attn_masks,
                decoder_start_token_id=decoder_start_token_id,
                **generate_params
            )
            generated_text = self.tokenizer.batch_decode(sample_outputs, **decode_params)
            generate_corpus = [text.split() for text in generated_text]
            return generate_corpus
