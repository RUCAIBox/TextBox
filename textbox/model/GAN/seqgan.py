# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

r"""
SeqGAN
################################################
Reference:
    Yu et al. "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient" in AAAI 2017.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.SeqGANGenerator import SeqGANGenerator
from textbox.module.Discriminator.SeqGANDiscriminator import SeqGANDiscriminator


class SeqGAN(GenerativeAdversarialNet):
    r"""SeqGAN is a generative adversarial network consisting of a generator and a discriminator.
        Modeling the data generator as a stochastic policy in reinforcement learning (RL), 
        SeqGAN bypasses the generator differentiation problem by directly performing gradient policy update.
        The RL reward signal comes from the GAN discriminator judged on a complete sequence,
        and is passed back to the intermediate state-action steps using Monte Carlo search.
    """

    def __init__(self, config, dataset):
        super(SeqGAN, self).__init__(config, dataset)
        self.generator = SeqGANGenerator(config, dataset)
        self.discriminator = SeqGANDiscriminator(config, dataset)
        self.pad_idx = dataset.padding_token_idx

    def calculate_g_train_loss(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus)

    def calculate_d_train_loss(self, real_data, fake_data, epoch_idx):
        return self.discriminator.calculate_loss(real_data, fake_data)

    def calculate_g_adversarial_loss(self, epoch_idx):
        self.discriminator.eval()
        loss = self.generator.adversarial_loss(self.discriminator.forward)
        self.discriminator.train()
        return loss

    def calculate_nll_test(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus, nll_test=True)

    def generate(self, eval_data):
        return self.generator.generate(eval_data)

    def sample(self, sample_num):
        samples = self.generator.sample(sample_num)
        return samples
