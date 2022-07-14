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
"""


import torch.nn as nn
import warnings
from .abstract_model import AbstractModel
from textbox import CLM_MODELS, SEQ2SEQ_MODELS

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, EncoderDecoderModel

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

class Pretrained_Models(AbstractModel):
    def __init__(self, config, tokenizer):
        super(Pretrained_Models, self).__init__(config, tokenizer)
        self.source_max_length = config['src_len']
        self.target_max_length = config['tgt_len']

        # check model
        self.model_name = config['model_name']
        model_path = config['model_path']
        self.is_casual_model = bool(self.model_name in CLM_MODELS)
        self.is_seq2seq_model = bool(self.model_name in SEQ2SEQ_MODELS)

        # loading config
        config_path = config['config_path'] or model_path
        config_kwargs = config['config_kwargs'] or {}
        self.configuration = AutoConfig.from_pretrained(config_path, **config_kwargs)

        self._init_params()
        # self._process_prompt()

        # loading model
        if self.model_name == 'bert2bert':
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path, model_path, config=self.configuration)
        elif self.is_casual_model:
            self.configuration.is_decoder = True
            if model_path:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, config=self.configuration)
            else:
                warnings.warn(f"Initialize {self.model_name} from scratch")
                self.model = AutoModelForCausalLM.from_config(self.configuration)
        else:
            if model_path:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=self.configuration)
            else:
                warnings.warn(f"Initialize {self.model_name} from scratch")
                self.model = AutoModelForSeq2SeqLM.from_config(self.configuration)

        if self.model_name != 'bert2bert':
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.label_smoothing = config['label_smoothing'] if config['label_smoothing'] else 0.

        # geneation settings
        self.generation_kwargs = {}
        self.generation_kwargs['max_length'] = self.target_max_length
        self.generation_kwargs['decoder_start_token_id'] = self.configuration.decoder_start_token_id if self.model_name != 'mbart' else self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
        self.generation_kwargs.update(config['generation_kwargs'] or {})

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
        pegasus: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        prophetnet: [src, [SEP]], [tgt, [SEP]], decoder_start_token_id: [SEP]
        t5, mt5: [src, </s>], [tgt, </s>], decoder_start_token_id: <pad>
        """
        # configuration needs to add pad token
        if self.model_name in ['ctrl', 'gpt2', 'gpt_neo', 'openai-gpt']:
            self.configuration.pad_token_id = self.tokenizer.eos_token_id
        
        # init `forced_bos_token_id` token
        if self.model_name in ['bart', 'led', 'mvp']:
            self.configuration.forced_bos_token_id = self.tokenizer.bos_token_id
        elif self.model_name == 'm2m_100':
            self.configuration.forced_bos_token_id = self.tokenizer.get_lang_id(self.config['tgt_lang'])
        
        # used in generate() for casual models
        if self.is_casual_model:
            self.configuration.eos_token_id = self.tokenizer.eos_token_id
        
        # special settings for cpm
        if self.model_name == 'cpm':
            import jieba
            import os
            jieba.dt.tmp_dir = "/tmp/jieba"
            os.makedirs(jieba.dt.tmp_dir, exist_ok=True)

    def forward(self, batch, epoch_idx=-1):
        inputs = {
            'input_ids': batch['source_ids'].to(self.device),
            'attention_mask': batch['source_mask'].to(self.device),
            'labels': batch['target_ids'].to(self.device)
        }
        outputs = self.model(**inputs)
        
        if self.label_smoothing:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            return loss_fct(outputs.logits.view(len(batch['source_text']), -1), inputs['labels'].view(-1))
        else:
            return outputs.loss

    def generate(self, batch, eval_data, accelerator):
        inputs = {
            'input_ids': batch['source_ids'].to(self.device),
            'attention_mask': batch['source_mask'].to(self.device),
        }
        
        if self.is_casual_model:
            input_ids_len = inputs['input_ids'].shape[1]
            self.generation_kwargs['max_length'] += input_ids_len
        
        # sample_outputs = self.model.generate(**inputs, **self.generation_kwargs)
        sample_outputs = accelerator.unwrap_model(self.model).generate(**inputs, **self.generation_kwargs)
        sample_outputs = accelerator.pad_across_processes(
            sample_outputs, dim=1, pad_index=self.tokenizer.pad_token_id
        )
        sample_outputs = accelerator.gather((sample_outputs))
        
        if self.is_casual_model:
            sample_outputs = sample_outputs[:, input_ids_len:]
        
        decode_kwargs = {'skip_special_tokens': True, 'clean_up_tokenization_spaces': True}
        generated_text = self.tokenizer.batch_decode(sample_outputs, **decode_kwargs)
        return generated_text
