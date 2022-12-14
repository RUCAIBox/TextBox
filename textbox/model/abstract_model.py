import torch
import torch.nn as nn
from textbox import CLM_MODELS, SEQ2SEQ_MODELS, RNN_MODELS, PLM_MODELS
from transformers import EncoderDecoderModel
import os
from typing import List, Optional, Tuple, Union
from transformers.modeling_utils import get_parameter_dtype
from collections import OrderedDict


class AbstractModel(nn.Module):
    r"""Base class for all models
    """

    def __init__(self, config, tokenizer):
        # load parameters info
        super(AbstractModel, self).__init__()
        self.device = config['device']
        self.config = config
        self.tokenizer = tokenizer
        self.source_max_length = config['src_len']
        self.target_max_length = config['tgt_len']

        # check model
        self.model_name = config['model_name']
        self.is_casual_model = bool(self.model_name in CLM_MODELS)
        self.is_seq2seq_model = bool(self.model_name in SEQ2SEQ_MODELS or self.model_name in RNN_MODELS)

        self.is_prompt_tuning = 'prompt-tuning' in config['efficient_methods']
        self.label_smoothing = config['label_smoothing'] if config['label_smoothing'] else 0.

    def generate_setting(self, config):
        # geneation settings
        self.generation_kwargs = {}
        self.generation_kwargs['max_length'] = self.target_max_length
        if self.model_name in PLM_MODELS:
            # transformer models
            self.generation_kwargs['decoder_start_token_id'] = \
                self.configuration.decoder_start_token_id \
                if self.model_name != 'mbart' else \
                self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
        self.generation_kwargs.update(config['generation_kwargs'] or {})

    def generate(self, batch_data):
        r"""Predict the texts conditioned on a noise or sequence.

        Args:
            batch_data (Corpus): Corpus class of a single batch.

        Returns:
            torch.Tensor: Generated text, shape: [batch_size, max_len]
        """
        raise NotImplementedError

    def _process_prompt_tuning_input(self, inputs, batch):
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, batch, epoch_idx=-1):
        inputs = self.process_forward_inputs(batch)

        if self.is_prompt_tuning:
            inputs = self._process_prompt_tuning_input(inputs, batch)
        outputs = self.model(**inputs)

        if self.label_smoothing:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            vocab_size = outputs.logits.size(-1)
            if self.is_casual_model:
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = inputs['labels'][..., 1:].contiguous()
            else:
                logits = outputs.logits
                labels = inputs['labels']
            return loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        else:
            return outputs.loss

    def generate(self, batch, accelerator):
        inputs = self.process_generate_inputs(batch)

        if self.is_prompt_tuning:
            inputs = self._process_prompt_tuning_input(inputs, batch)

        if self.is_casual_model:
            input_ids_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else inputs['inputs_embeds'].shape[1]
            self.generation_kwargs['max_length'] = self.target_max_length + input_ids_len

        # sample_outputs = self.model.generate(**inputs, **self.generation_kwargs)
        sample_outputs = accelerator.unwrap_model(self.model).generate(**inputs, **self.generation_kwargs)
        sample_outputs = accelerator.pad_across_processes(sample_outputs, dim=1, pad_index=self.tokenizer.pad_token_id)
        sample_outputs = accelerator.gather((sample_outputs))

        if self.is_casual_model:
            sample_outputs = sample_outputs[:, input_ids_len:]

        decode_kwargs = {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}
        generated_text = self.tokenizer.batch_decode(sample_outputs, **decode_kwargs)
        generated_text = [g.strip() or 'NULL' for g in generated_text]
        return generated_text

    def process_forward_inputs(self, batch):
        inputs = self.process_generate_inputs(batch)
        inputs.update({'labels': batch['target_ids'].to(self.device)})
        return inputs

    def process_generate_inputs(self, batch):
        inputs = {
            'input_ids': batch['source_ids'].to(self.device),
            'attention_mask': batch['source_mask'].to(self.device),
        }
        return inputs

    def from_pretrained(self, save_directory: Union[str, os.PathLike]):
        if self.model_name in ['bert2bert', 'xlm-roberta', 'xlm']:
            self.model = EncoderDecoderModel.from_pretrained(save_directory)
        else:
            model_path = os.path.join(save_directory, 'pytorch_model.bin')
            model_load = torch.load(model_path, map_location=self.device)
            self.load_state_dict(model_load)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
    ):
        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(self)
        self.configuration.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        self.configuration.architectures = [self.model.__class__.__name__]

        # Save the config
        if is_main_process:
            self.configuration.save_pretrained(save_directory)

        # Save the tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        if self.model_name in ['bert2bert', 'xlm-roberta', 'xlm']:
            self.model.save_pretrained(save_directory)
        else:
            state_dict = OrderedDict([(k, v.detach().cpu()) for k, v in self.state_dict().items()])
            torch.save(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))
