# @Time   : 2020/12/26
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn


import torch
import numpy as np

from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.MaskGANGenerator import MaskGANGenerator
from textbox.module.Discriminator.MaskGANDiscriminator import MaskGANDiscriminator


class MaskGAN(GenerativeAdversarialNet):
    r""" MaskGan: better text generation via filling in the mask.
    """

    def __init__(self, config, dataset):
        super(MaskGAN, self).__init__(config, dataset)
        self.source_vocab_size = self.vocab_size
        self.target_vocab_size = self.vocab_size
        self.generator = MaskGANGenerator(config, dataset)
        self.discriminator = MaskGANDiscriminator(config, dataset)
        self.pad_idx = dataset.padding_token_idx
        self.eos_idx = dataset.eos_token_idx
        self.mask_strategy = config['mask_strategy']
        self.is_present_rate = config['is_present_rate']
        self.is_present_rate_decay = config['is_present_rate_decay']
        self.max_length = config['max_seq_length']

    def calculate_g_train_loss(self, corpus, epoch_idx=0, validate=False):
        r""" Specified for maskgan calculate generator masked token predicted
        """
        real_inputs = corpus[:, :-1]  # bs * self.max_len - 1
        target_inputs = corpus[:, 1:]
        bs, seq_len = target_inputs.size()
        lengths = torch.tensor([seq_len] * bs)
        target_present = self.generate_mask(bs, seq_len, "continuous")
        device = target_inputs.device
        lengths = lengths.cuda(device)
        target_present = target_present.cuda(device)
        return self.generator.calculate_train_loss(real_inputs, lengths, target_inputs, target_present,
                                                   validate=validate)

    def calculate_d_train_loss(self, data, epoch_idx):
        r""" Specified for maskgan calculate discriminator masked token predicted
        """
        self.generator.eval()
        inputs = data[:, :-1]
        targets = data[:, 1:]
        batch_size, seq_len = inputs.size()
        lengths = torch.tensor([seq_len] * batch_size)
        targets_present = self.generate_mask(batch_size, seq_len, "continuous")
        device = inputs.device
        targets_present = targets_present.cuda(device)
        lengths = lengths.cuda(device)

        fake_sequence, _, _ = self.generator.forward(inputs, lengths, targets, targets_present)
        self.generator.train()
        return self.discriminator.calculate_loss(inputs, lengths, fake_sequence, targets_present,
                                                 self.generator.embedder)

    def generate_mask(self, batch_size, seq_len, mask_strategy):
        r"""Generate the mask to be fed into the model.
        """
        if mask_strategy == 'random':
            p = np.random.choice(
                [True, False],
                size=[batch_size, seq_len],
                p=[self.is_present_rate, 1. - self.is_present_rate])
        elif mask_strategy == 'continuous':
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

    def calculate_g_adversarial_loss(self, data, epoch_idx):
        r""" Specified for maskgan calculate adversarial masked token predicted
        """
        real_inputs = data[:, :-1]
        target_inputs = data[:, 1:]
        batch_size, seq_len = real_inputs.size()
        lengths = torch.tensor([seq_len] * batch_size)
        targets_present = self.generate_mask(batch_size, seq_len, "continuous")
        device = real_inputs.device
        targets_present = targets_present.cuda(device)
        lengths = lengths.cuda(device)

        loss = self.generator.adversarial_loss(real_inputs, lengths, target_inputs, targets_present, self.discriminator)
        return loss

    def calculate_nll_test(self, eval_batch, epoch_idx):
        r""" Specified for maskgan calculating the negative log-likelihood of the batch.
        """
        real_inputs = eval_batch[:, :-1]
        target_inputs = eval_batch[:, 1:]
        batch_size, seq_len = real_inputs.size()
        lengths = torch.tensor([seq_len] * batch_size)
        targets_present = torch.zeros_like(target_inputs).byte()
        device = real_inputs.device
        lengths = lengths.cuda(device)
        outputs, log_probs, logits = self.generator.forward(real_inputs, lengths, target_inputs, targets_present)
        return self.generator.calculate_loss(logits, target_inputs)

    def generate(self, eval_data):
        return self.generator.generate(eval_data)

    def update_is_present_rate(self):
        self.is_present_rate *= (1. - self.is_present_rate_decay)
