r"""
################################################
    Integration of Encoder-Decoder LM and Casual LM in transformers API

    Summary of Related Papers:
    gpt2: Language Models are Unsupervised Multitask Learners
    openai-gpt: Improving Language Understanding by Generative Pre-Training
    cpm: CPM: A Large-scale Generative Chinese Pre-trained Language Model
    ctrl: CTRL: A Conditional Transformer Language Model for Controllable Generation
    gpt_neo: The Pile: An 800GB Dataset of Diverse Text for Language Modeling

    t5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    mt5: mT5: A massively multilingual pre-trained text-to-text transformer
    bart: BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
    led: Longformer: The Long-Document Transformer
    mbart: Multilingual Denoising Pre-training for Neural Machine Translation
    bert2bert: Leveraging Pre-trained Checkpoints for Sequence Generation Tasks
    pegasus: PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
    blenderbot, blenderbot-small: Recipes for building an open-domain chatbot
    m2m_100: Beyond English-Centric Multilingual Machine Translation
    prophetnet: ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training
"""


import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from torch.nn.utils.rnn import pad_sequence
from .abstract_model import AbstractModel
from textbox import CLM_MODELS, SEQ2SEQ_MODELS

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, EncoderDecoderModel

'''
# Model for Causal LM mapping
("xlm", "XLMWithLMHeadModel"),
("xlm-roberta", "XLMRobertaForCausalLM"),
("xlnet", "XLNetLMHeadModel"),

# Model for Seq2Seq Causal LM mapping
("longt5", "LongT5ForConditionalGeneration"),
("marian", "MarianMTModel"),
("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
'''

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
    padded_tensors = torch.stack(padded_tensors, dim=0)
    return padded_tensors

class Pretrained_Models(AbstractModel):
    def __init__(self, config, dataset):
        super(Pretrained_Models, self).__init__(config, dataset)
        self.config = config

        # Check model
        self.model_name = config['model'].lower()
        self.model_path = config['model_path']
        self.is_casual_model = bool(self.model_name in CLM_MODELS)
        self.is_seq2seq_model = bool(self.model_name in SEQ2SEQ_MODELS)
        assert self.is_casual_model ^ self.is_seq2seq_model, f"model '{self.model_name}' must be either CLMs or Seq2SeqLMs"

        # Loading config
        self.config_path = config['config_path'] or self.model_path
        config_kwargs = config['config_kwargs'] or {}
        self.configuration = AutoConfig.from_pretrained(self.config_path, **config_kwargs)

        # Loading tokenizer
        tokenizer_kwargs = config['tokenizer_kwargs'] or {}
        self.tokenizer_path = config['tokenizer_path'] or self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, **tokenizer_kwargs)
        self.truncate = config['truncate'] or 'tail'

        # Loading model
        if self.model_name == 'bert2bert':
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.model_name, self.model_name)
        elif self.is_casual_model:
            self.configuration.is_decoder = True
            if self.model_path:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, config=self.configuration)
            else:
                self.model = AutoModelForCausalLM.from_config(self.configuration)
        else:
            if self.model_path:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, config=self.configuration)
            else:
                self.model = AutoModelForSeq2SeqLM.from_config(self.configuration)

        self._init_params()
        self._process_prompt()
        if self.model_name != 'bert2bert':
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.label_smoothing = config['label_smoothing'] if config['label_smoothing'] else 0.
        self.generation_kwargs = config['generation_kwargs']

    def _init_params(self):
        # (1): tokenizer needs to add eos token
        if self.model_name in ['ctrl', 'openai-gpt']:
            self.tokenizer.add_special_tokens(({'eos_token': '</s>'}))

        # (2): tokenizer needs to add pad token
        if self.model_name in ['ctrl', 'gpt2', 'gpt_neo', 'openai-gpt']:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # (3): tokenizer needs to change replace eos token with sep token
        if self.model_name in ['cpm']:
            self.tokenizer.eos_token = self.tokenizer.sep_token

        # (4): tokenizer needs to modify `build_inputs_with_special_tokens()` and `num_special_tokens_to_add()`
        if self.model_name in ['blenderbot-small', 'cpm', 'ctrl', 'gpt2', 'gpt_neo', 'openai-gpt']:
            self.tokenizer.build_inputs_with_special_tokens = lambda t0, t1=None: t0 + [self.tokenizer.eos_token_id]
            self.tokenizer.num_special_tokens_to_add = lambda : 1
        elif self.model_name in ['opt']:
            self.tokenizer.build_inputs_with_special_tokens = lambda t0, t1=None: [self.tokenizer.bos_token_id] + t0 + [self.tokenizer.eos_token_id]
            self.tokenizer.num_special_tokens_to_add = lambda : 2

        # (5): tokenizer needs to set src_lang, tgt_lang (used in translation task)
        if self.model_name in ['m2m_100', 'mbart']:
            assert self.config['src_lang'] and self.config['tgt_lang'], \
                self.model_name + ' needs to specify source language and target language with `--src_lang=xx` and `--tgt_lang=xx`'
            self.tokenizer.src_lang = self.config['src_lang']
            self.tokenizer.tgt_lang = self.config['tgt_lang']

        # (6): model specific init
        if self.model_name in ['bart', 'led', 'mvp']:
            self.configuration.forced_bos_token_id = self.tokenizer.bos_token_id
        elif self.model_name == 'm2m_100':
            self.configuration.forced_bos_token_id = self.tokenizer.get_lang_id(self.config['tgt_lang'])
        elif self.model_name == 'bert2bert':
            self.configuration.decoder_start_token_id = self.tokenizer.cls_token_id
            self.configuration.pad_token_id = self.tokenizer.pad_token_id
        
        # used in generate() for casual models
        if self.is_casual_model:
            self.configuration.eos_token_id = self.tokenizer.eos_token_id

    def _process_prompt(self):
        r"""
        Prompts can be added to the beginning and end of **source ids**.
        """
        self.prefix = self.config['prefix_prompt'] if self.config['prefix_prompt'] else ''
        self.suffix = self.config['suffix_prompt'] if self.config['suffix_prompt'] else ''

        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        self.src_ids_num = self.source_max_length - self.tokenizer.num_special_tokens_to_add() - len(self.prefix_ids) - len(self.suffix_ids)
        self.tgt_ids_num = self.target_max_length - self.tokenizer.num_special_tokens_to_add()

        if self.model_name in ['bert2bert', 'opt']:
            self.tgt_ids_num += 1

    def _generate_default_inputs(self, corpus):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        input_ids = []
        labels = []
        attn_masks = []

        pad_token_id = self.tokenizer.pad_token_id
        padding_side = self.tokenizer.padding_side

        for src, tgt in zip(source_text, target_text):
            input_id, label = self._model_encode(src, tgt_text=tgt)

            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            attn_masks.append(torch.ones(len(input_id), dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))

        input_ids = _pad_sequence(input_ids, padding_value=pad_token_id, padding_side=padding_side).to(self.device)
        attn_masks = _pad_sequence(attn_masks, padding_value=0, padding_side=padding_side).to(self.device)
        labels = _pad_sequence(labels, padding_value=-100, padding_side=padding_side).to(self.device)

        inputs = {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}
        return inputs

    def _model_encode(self, src_text, tgt_text=None, eval=False):
        r"""
        Casual models:
        cpm: [src, <sep>; tgt, <sep>]
        ctrl, openai-gpt: [src, </s>; tgt, </s>]
        gpt2, gpt_neo: [src, <|endoftext|>; tgt, <|endoftext|>]
        opt: [</s>, src, </s>; tgt, </s>]

        Encoder-decoder models:
        bart, led, mvp: [<s>, src, </s>], [<s>, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: <s>
        bert2bert: [[CLS], src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [CLS]
        bigbird_pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        blenderbot: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        blenderbot-small: [src, __end__], [tgt, __end__], decoder_start_token_id: __start__
        m2m_100: [src_lang_id, src, </s>], [tgt_lang_id, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: tgt_lang_id
        mbart: [src, </s>, src_lang_id], [tgt, </s>, tgt_lang_id], decoder_start_token_id: tgt_lang_id
        pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        prophetnet: [src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [SEP]
        t5, mt5: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        """

        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        src_ids = src_ids[:self.src_ids_num] if self.truncate == 'tail' else src_ids[-self.src_ids_num:]
        src_ids = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + src_ids + self.suffix_ids)
        if eval:
            return src_ids

        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=False)
        tgt_ids = tgt_ids[:self.tgt_ids_num]
        with self.tokenizer.as_target_tokenizer():
            tgt_ids = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)
        
        # model specific process, remove decoder bos_token
        if self.model_name in ['bert2bert', 'opt']: 
            tgt_ids = tgt_ids[1:]

        if self.is_casual_model:
            input_id = src_ids + tgt_ids
            label = len(src_ids) * [-100] + tgt_ids
        else:
            input_id = src_ids
            label = tgt_ids
        return input_id, label

    def forward(self, corpus, epoch_idx=-1):
        batch_size = len(corpus['source_text'])
        inputs = self._generate_default_inputs(corpus)
        outputs = self.model(**inputs)
        if self.label_smoothing:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            return loss_fct(outputs.logits.view(batch_size, -1), inputs['labels'].view(-1))
        else:
            return outputs.loss

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']

        pad_token_id = self.tokenizer.pad_token_id
        padding_side = self.tokenizer.padding_side
        decode_params = {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}
        self.generation_kwargs['max_length'] = self.target_max_length

        if self.is_casual_model:
            input_ids = []
            attn_masks = []
            for src in source_text:
                input_id = self._model_encode(src, eval=True)
                input_ids.append(torch.tensor(input_id, dtype=torch.long))
                attn_masks.append(torch.ones(len(input_id), dtype=torch.long))

            input_ids = _pad_sequence(input_ids, padding_value=pad_token_id, padding_side='left').to(self.device)
            attn_masks = _pad_sequence(attn_masks, padding_value=0, padding_side='left').to(self.device)
            input_id_len = input_ids.shape[1]
            self.generation_kwargs['max_length'] += input_id_len
            sample_outputs = self.model.generate(input_ids, attention_mask=attn_masks, **self.generation_kwargs)
            generated_text = self.tokenizer.batch_decode(sample_outputs[:, input_id_len:], **decode_params)
            generate_corpus = [text.split() for text in generated_text]
            return generate_corpus

        else:
            input_ids = [torch.tensor(self._model_encode(src, eval=True), dtype=torch.long)
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
                **self.generation_kwargs
            )
            generated_text = self.tokenizer.batch_decode(sample_outputs, **decode_params)
            generate_corpus = [text.split() for text in generated_text]
            return generate_corpus
