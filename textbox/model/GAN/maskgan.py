# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from textbox.utils import InputType
from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.MaskGANGenerator import MaskGANGenerator
from textbox.module.Discriminator.MaskGANDiscriminator import MaskGANDiscriminator


class MaskGAN(GenerativeAdversarialNet):
    """Sequence Generative Adversarial Nets with Policy Gradient

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(MaskGAN, self).__init__(config, dataset)
        self.source_vocab_size = self.vocab_size
        self.target_vocab_size = self.vocab_size
        self.generator = MaskGANGenerator(config, dataset)
        self.discriminator = MaskGANDiscriminator(config, dataset)
        self.pad_idx = dataset.padding_token_idx
        self.mask_strategy = config['mask_strategy']
        self.is_present_rate = config['is_present_rate']

    def calculate_g_train_loss(self, corpus, is_advtrain, epoch_idx):
        r"""For pretraining with cross entropy loss, we have all tokens in the forward sequence present (all True).

        Args:
            corpus:
            epoch_idx:
            is_advtrain:

        Returns:

        """
        real_inputs = corpus['target_idx'][:, :-1]
        target_inputs = corpus['target_idx'][:, 1:]
        # lengths = self.get_length(real_inputs)
        lengths = corpus['target_length']-1
        if not is_advtrain:
            target_present = torch.ones_like(target_inputs, dtype=torch.bool)
        else:
            batch_size, seq_len = target_inputs.size()
            target_present = self.generate_mask(batch_size, seq_len)
            target_present = target_present.cuda(self.device)
        return self.generator.calculate_train_loss(real_inputs, lengths, target_inputs, target_present, is_advtrain)

    def calculate_d_train_loss(self, data, epoch_idx, is_advtrain=True):
        self.generator.eval()
        inputs = data['target_idx'][:, :-1]
        targets = data['target_idx'][:, 1:]
        # lengths = self.get_length(inputs)
        lengths = data['target_length'] - 1
        batch_size, seq_len = inputs.size()
        targets_present = self.generate_mask(batch_size, seq_len)
        targets_present = targets_present.cuda(self.device)
        fake_sequence, _, _ = self.generator.forward(inputs, lengths, targets, targets_present, is_advtrain=is_advtrain)  # is_advtrain?
        self.generator.train()
        return self.discriminator.calculate_loss(inputs, lengths, fake_sequence, targets_present)

    def generate_mask(self, batch_size, seq_len):
        """Generate the mask to be fed into the model."""
        if self.mask_strategy == 'random':
            p = np.random.choice(
                [True, False],
                size=[batch_size, seq_len],
                p=[self.is_present_rate, 1. - self.is_present_rate])
        elif self.mask_strategy == 'contiguous':
            masked_length = int((1 - self.is_present_rate) * seq_len) - 1
            # Determine location to start masking.
            start_mask = np.random.randint(1, seq_len - masked_length + 1, size=batch_size)
            p = np.full([batch_size, seq_len], True, dtype=bool)

            # Create contiguous masked section to be False.
            for i, index in enumerate(start_mask):
                p[i, index:index + masked_length] = False
        else:
            raise NotImplementedError
        p = torch.from_numpy(p)

        return p

    def get_length(self, inputs):
        length_list = []
        batch_size, seq_len = inputs.size()
        for i in range(batch_size):
            sentence = inputs[i, :].cpu().numpy().tolist()
            for j in range(seq_len):
                if sentence[j] == self.pad_idx:
                    length_list.append(j)
                    break
        length_list = torch.Tensor(length_list)
        device = inputs.device
        length_list.cuda(device)
        return length_list

    def calculate_g_adversarial_loss(self, data, epoch_idx):
        real_inputs = data['target_idx'][:, :-1]
        target_inputs = data['target_idx'][:, 1:]
        lengths = data['target_length'] - 1
        batch_size, seq_len = target_inputs.size()
        target_present = self.generate_mask(batch_size, seq_len)
        target_present = target_present.cuda(self.device)
        loss = self.generator.adversarial_loss(real_inputs, lengths, target_inputs, target_present, self.discriminator)
        return loss

    def calculate_loss(self, corpus):
        pass

    def generate(self, eval_data):
        return self.generator.generate(eval_data)

    def sample(self, sample_num):
        pass
