import torch
from torch import Tensor
from typing import List

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
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel,
    MegatronBertForCausalLM,
    XLNetTokenizer, XLNetLMHeadModel,
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
    'led': {
        'tokenizer': LEDTokenizer,
        'model': LEDForConditionalGeneration
    },
    'm2m100': {
        'tokenizer': M2M100Tokenizer,
        'model': M2M100ForConditionalGeneration
    },
    'mbart': {
        'tokenizer': MBartTokenizer,
        'model': MBartForConditionalGeneration
    },
    'prophetnet': {
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
    'xlnet': {
        'tokenizer': XLNetTokenizer,
        'model': XLNetLMHeadModel
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

CLM_MODELS = ['gpt2', 'gpt', 'big_bird', 'bert', 'roberta', 'cpm', 'ctrl', 'xlnet', 'megatron_bert', 'transfo_xl',
              'gpt_neo']

EncDecLM_MODELS = ['t5', 'mt5', 'bart', 'mbart', 'bert2bert', 'big_bird_pegasus', 'pegasus', 'blender_bot',
                   'blender_bot_small', 'led', 'm2m100', 'prophetnet']


def pad_sequence(tensors: List[Tensor], padding_value: int, padding_side: str = 'right'):
    """
    Pad encoded inputs (on left/right and up to max length in the batch)
    """
    max_len = max(tensor.size()[0] for tensor in tensors)
    padded_tensors = []
    if padding_side == 'right':
        for tensor in tensors:
            padding_length = max_len-len(tensor)
            padded_tensor = torch.cat([tensor, torch.full([padding_length], padding_value, dtype=tensor.dtype)], dim=-1)
            padded_tensors.append(padded_tensor)
    elif padding_side == 'left':
        for tensor in tensors:
            padding_length = max_len-len(tensor)
            padded_tensor = torch.cat([torch.full([padding_length], padding_value, dtype=tensor.dtype), tensor], dim=-1)
            padded_tensors.append(padded_tensor)
    else:
        raise ValueError("Invalid padding strategy:" + str(padding_side))
    padded_tensors = torch.tensor(padded_tensors)
    return padded_tensors


class Transformers(Seq2SeqGenerator):
    def __init__(self, config, dataset):
        super(Transformers, self).__init__(config, dataset)

        self.model_name_or_path = config['pretrained_model_path']
        self.model_name = config['model'].lower()
        self.is_casual_model = bool(self.model_name in CLM_MODELS)
        self.is_enc_dec_model = bool(self.model_name in EncDecLM_MODELS)
        assert self.is_casual_model or self.is_enc_dec_model, "model must be one of CLMs or EncDecLMs"

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

    def _process_prompt(self):
        """
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
        if self.model_name in ['gpt2', 'transfo_xl', 'gpt_neo']:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # (3): tokenizer needs to modify build_inputs_with_special_tokens()
        if self.model_name in ['gpt2', 'transfo_xl', 'blender_bot_small', 'gpt_neo']:
            self.tokenizer.build_inputs_with_special_tokens = lambda t0, t1: t0 + [self.tokenizer.eos_token_id]

        # (4): tokenizer needs to set src_lang, tgt_lang (used in translation task)
        if self.model_name in ['m2m100', 'mbart']:
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
        BOS = cls_token: big_bird, bert, roberta, cpm, megatron_bert, xlnet
            = bos_token: gpt2, gpt_neo
            = None: ctrl, gpt, transfo_xl

        EOS = sep_token: big_bird, bert, roberta, cpm, megatron_bert, xlnet
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

    def _generate_default_inputs(self, corpus):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        input_ids = []
        labels = []
        attn_masks = []

        pad_token_id = self.tokenizer.pad_token_id
        padding_side = self.tokenizer.padding_side

        for src, tgt in zip(source_text, target_text):
            src_ids = self.tokenizer.encode(src, add_special_tokens=False)
            tgt_ids = self.tokenizer.encode(tgt, add_special_tokens=False)

            if self.is_casual_model:
                input_id, label = self._casual_model_encode(src_ids, tgt_ids)
            else:
                input_id, label = self._encoder_decoder_model_encode(src_ids, tgt_ids)

            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            attn_masks.append(torch.ones(len(input_id), dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))

        input_ids = pad_sequence(input_ids, padding_value=pad_token_id, padding_side=padding_side).to(self.device)
        attn_masks = pad_sequence(attn_masks, padding_value=0, padding_side=padding_side).to(self.device)
        labels = pad_sequence(labels, padding_value=-100, padding_side=padding_side).to(self.device)

        inputs = {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}
        processed_inputs = self._inputs_postprocess(**inputs)
        return processed_inputs

    def _encoder_decoder_model_encode(self, src_ids, tgt_ids):
        """
        t5, mt5: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        bart, led: [<s>, src, </s>], [<s>, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: <s>
        bert2bert: [[CLS], src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [CLS]
        big_bird_pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        blender_bot: [src, </s>], [tgt, </s>], decoder_start_token_id: <s>
        blender_bot_small: [src, __end__], [tgt, __end__], decoder_start_token_id: __start__
        m2m100: [src_lang_id, src, </s>], [tgt_lang_id, tgt, </s>], decoder_start_token_id: </s>, forced_bos_token_id: tgt_lang_id
        mbart: [src, </s>, src_lang_id], [tgt, </s>, tgt_lang_id], decoder_start_token_id: tgt_lang_id
        """
        src_ids = src_ids[:self.source_max_length - self.tokenizer.num_special_tokens_to_add()
                          - len(self.prefix_ids) - len(self.suffix_ids)]
        tgt_ids = tgt_ids[:self.target_max_length - self.tokenizer.num_special_tokens_to_add()]
        input_id = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + src_ids + self.suffix_ids)
        if self.model_name in ['m2m100', 'mbart']:
            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)
        else:
            label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)

        return input_id, label

    def _casual_model_encode(self, src_ids, tgt_ids):
        """
        gpt2, gpt_neo: [<|endoftext|>, src, <|endoftext|>, tgt, <|endoftext|>]
        big_bird, bert, megatron_bert: [[CLS], src, [SEP], tgt, [SEP]]
        roberta: [<s>, src, </s>, tgt, </s>]
        cpm, xlnet: [src, <sep>, tgt, <sep>, <cls>]
        ctrl, gpt: [src, </s>, tgt, </s>]
        transfo_xl: [src, <eos>, tgt, <eos>]
        """

        src_ids = src_ids[:self.source_max_length-len(self.prefix_ids)-len(self.suffix_ids)-1-len(self.bos_token_id)]
        tgt_ids = tgt_ids[:self.target_max_length - 1]

        src_input_id = self.prefix_ids + src_ids + self.suffix_ids + self.eos_token_id
        tgt_input_id = tgt_ids + self.eos_token_id

        if self.tokenizer.padding_side == 'left':  # cpm, xlnet
            tgt_input_id = tgt_input_id + self.bos_token_id
        else:
            src_input_id = self.bos_token_id + src_input_id

        input_id = src_input_id + tgt_input_id
        label = len(src_input_id) * [-100] + tgt_input_id

        return input_id, label

    def _inputs_postprocess(self, input_ids, attention_mask, labels):
        pass

    def forward(self, corpus, epoch_idx=-1):
        inputs = self._generate_default_inputs(corpus)
        outputs = self.model(**inputs)
        if hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            return outputs.losses.mean()

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