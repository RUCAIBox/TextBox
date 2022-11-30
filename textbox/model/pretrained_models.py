r"""
################################################
    Integration of casual models and encoder-decoder models in Transformers API

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
    unilm: Unified Language Model Pre-training for Natural Language Understanding and Generation
    mass: MASS: Masked Sequence to Sequence Pre-training for Language Generation
"""
import os
import copy
import torch
import torch.nn as nn
import warnings
import inspect
from .abstract_model import AbstractModel

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, EncoderDecoderModel
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from transformers.models.cpt.modeling_cpt import CPTForConditionalGeneration
from transformers.models.unilm.modeling_unilm import UnilmConfig, UnilmForSeq2Seq
from transformers.models.mass.modeling_mass import MassForConditionalGeneration
from transformers.models.mass.configuration_mass import MassConfig
from ..utils.argument_list import efficient_kwargs_dict
'''
# Model for Causal LM mapping
("xlm", "XLMWithLMHeadModel"),
("xlm-roberta", "XLMRobertaForCausalLM"),
("xlnet", "XLNetLMHeadModel"),

# Model for Seq2Seq Causal LM mapping
("longt5", "LongT5ForConditionalGeneration"),
("marian", "MarianMTModel"),
("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
("nllb", "M2M100ForConditionalGeneration"),
'''


class Pretrained_Models(AbstractModel):

    def __init__(self, config, tokenizer):
        super(Pretrained_Models, self).__init__(config, tokenizer)
        model_path = config['model_path']

        # loading config
        config_path = config['config_path'] or model_path or None
        config_kwargs = config['config_kwargs'] or {}
        if config_path is None:
            # No pretrained config. loading config from yaml
            model_type = config["model_type"]
            _name_or_path = config["_name_or_path"]
            params_list = (list(inspect.signature(AutoConfig.for_model(model_type).__init__).parameters.keys()))
            config_dict = {key: val for key, val in config.final_config_dict.items() if key in params_list}
            config_dict['model_type'] = model_type
            config_dict['_name_or_path'] = _name_or_path
            self.configuration = AutoConfig.for_model(**config_dict)
        else:
            # loading config from config_path
            if self.model_name == "unilm":
                config_kwargs["label_smoothing"] = self.label_smoothing
                self.label_smoothing = 0.
                self.configuration = UnilmConfig.from_pretrained(config_path, **config_kwargs)
            elif self.model_name == "mass":
                self.configuration = MassConfig.from_pretrained(config_path, **config_kwargs)
            else:
                self.configuration = AutoConfig.from_pretrained(config_path, **config_kwargs)
        if config['efficient_methods']:
            hard_efficient_methods = [
                m for m in ['prefix-tuning', 'p-tuning-v2', 'adapter', 'lora'] if m in config['efficient_methods']
            ]
            if hard_efficient_methods and self.model_name not in ['bart', 'gpt2', 't5']:
                raise NotImplementedError(
                    f'{self.model_name} does not currently support {hard_efficient_methods} method.'
                )
            self.configuration.efficient_methods = config['efficient_methods']
            efficient_kwargs = {}
            for method in config['efficient_methods']:
                efficient_kwargs.update(efficient_kwargs_dict[method])
            if config['efficient_kwargs']:
                efficient_kwargs.update(config['efficient_kwargs'])
            self.configuration.update(efficient_kwargs)

        self._init_params()

        # loading model
        model_class = None
        if self.model_name in ['bert2bert', 'xlm-roberta', 'xlm']:
            model_class = EncoderDecoderModel
        elif self.model_name == 'cpt':
            model_class = CPTForConditionalGeneration
        elif self.model_name == "unilm":
            model_class = UnilmForSeq2Seq
        elif self.model_name == 'mass':
            model_class = MassForConditionalGeneration
        elif self.is_casual_model:
            model_class = AutoModelForCausalLM
            self.configuration.is_decoder = True
        else:
            model_class = AutoModelForSeq2SeqLM

        if model_path:
            if model_path.startswith(config['saved_dir']):
                self.model = model_class.from_config(self.configuration)
                path_to_load = os.path.join(model_path, 'pytorch_model.bin')
                model_load = torch.load(path_to_load, map_location=self.device)
                self.load_state_dict(model_load)
                del model_load
            else:
                if self.model_name in ['bert2bert', 'xlm-roberta', 'xlm']:
                    self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                        model_path, model_path, config=self.configuration
                    )
                else:
                    self.model = model_class.from_pretrained(model_path, config=self.configuration)
        else:
            warnings.warn(f"Initialize {self.model_name} from scratch")
            self.model = model_class.from_config(self.configuration)

        if self.model_name == 'unilm':
            mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])
            self.model.additional_init(mask_word_id, eos_word_ids, sos_word_id)

        if self.model_name not in ['bert2bert', 'unilm', 'xlm-roberta', 'xlm']:
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif self.model_name not in ['unilm']:
            self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.is_prompt_tuning:
            self.prompt_length = self.model.config.prompt_length
            self.prompt_embedding = nn.Embedding(self.prompt_length, self.model.config.hidden_size)

        if config['lightweight_tuning']:
            self.model.set_lightweight_tuning()

        if config['efficient_methods'] and not config['efficient_unfreeze_model']:
            if hard_efficient_methods:
                self.model.set_efficient_tuning()
            elif 'bitfit' in config['efficient_methods']:
                for para in self.model.parameters():
                    if len(para.shape) > 1:
                        para.requires_grad_(False)
            elif 'prompt-tuning' in config['efficient_methods']:
                self.model.requires_grad_(False)
        self.generate_setting(config)

    def _init_params(self):
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
        pegasus, pegasus_x: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        prophetnet: [src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [SEP]
        t5, mt5: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        LongT5: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        marian: [src, </s>], [tgt, </s>], decoder_start_token_id: </s>
        xlm-roberta: [<s>, src, </s>; </s>, tgt, </s>], decoder_start_token_id: </s>
        xlm-prophetnet: [src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [SEP]
        nllb: [src_lang_id, src, </s>], [tgt_lang_id, tgt, </s>], decoder_start_token_id: </s>
        unilm: [[CLS], src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [SEP]
        mass: [src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [SEP]
        """
        # configuration needs to add pad token
        if self.model_name in ['ctrl', 'gpt2', 'gpt_neo', 'openai-gpt']:
            self.configuration.pad_token_id = self.tokenizer.eos_token_id

        # init `forced_bos_token_id` token
        if self.model_name in ['bart', 'led', 'mvp', 'mass']:
            self.configuration.forced_bos_token_id = self.tokenizer.bos_token_id
        elif self.model_name == 'm2m_100':
            self.configuration.forced_bos_token_id = self.tokenizer.get_lang_id(self.config['tgt_lang'])

        # used in generate() for casual models
        if self.is_casual_model:
            self.configuration.eos_token_id = self.tokenizer.eos_token_id

        # set lang_id for xlm
        if self.model_name == 'xlm':
            encoder_config = copy.deepcopy(self.configuration)
            decoder_config = copy.deepcopy(self.configuration)
            encoder_config.lang_id = self.configuration.lang2id[self.config['src_lang']]
            decoder_config.is_encoder = False
            decoder_config.causal = True
            decoder_config.lang_id = self.configuration.lang2id[self.config['tgt_lang']]
            self.configuration = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

        # special settings for cpm
        if self.model_name == 'cpm':
            import jieba
            import os
            jieba.dt.tmp_dir = "/tmp/jieba"
            os.makedirs(jieba.dt.tmp_dir, exist_ok=True)

    def _process_prompt_tuning_input(self, inputs, batch):
        input_ids = inputs['input_ids']
        inputs_embeds = self.model.get_input_embeddings()(input_ids)  # b, l, e
        prompt_embeds = self.prompt_embedding.weight.repeat(input_ids.size(0), 1, 1)  # b, pl, e
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        inputs['inputs_embeds'] = inputs_embeds
        del inputs['input_ids']
        mask = torch.ones(input_ids.size(0), self.prompt_length, dtype=torch.long).to(self.device)
        inputs['attention_mask'] = torch.cat([mask, inputs['attention_mask']], dim=1)
        if self.is_casual_model and 'labels' in inputs:
            labels = torch.full_like(mask, -100)
            inputs['labels'] = torch.cat([labels, inputs['labels']], dim=1)
        return inputs

    def process_forward_inputs(self, batch):
        if self.model_name != 'unilm':
            return super(Pretrained_Models, self).process_forward_inputs(batch)
        inputs = {
            'input_ids': batch['source_ids'].to(self.device),
            'attention_mask': batch['source_mask'].to(self.device),
            'token_type_ids': batch['token_type_ids'].to(self.device),
            'masked_lm_labels': batch['masked_lm_labels'].to(self.device),
            'masked_pos': batch['masked_pos'].to(self.device),
            'masked_weights': batch['masked_weights'].to(self.device),
        }
        return inputs

    def process_generate_inputs(self, batch):
        if self.model_name != 'unilm':
            return super(Pretrained_Models, self).process_generate_inputs(batch)
        inputs = {
            'input_ids': batch['source_ids'].to(self.device),
            'attention_mask': batch['source_mask'].to(self.device),
            'token_type_ids': batch['token_type_ids'].to(self.device),
            'position_ids': batch['position_ids'].to(self.device),
        }
        return inputs
