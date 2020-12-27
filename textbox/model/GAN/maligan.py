# @Time   : 2020/11/17
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn

r"""
MaliGAN
################################################
Reference:
    Tong Che et al. "Maximum-Likelihood Augmented Discrete Generative Adversarial Networks." in NIPS 2017.
Reference code:
    https://github.com/williamSYSU/TextGAN-PyTorch

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.MaliGANGenerator import MaliGANGenerator
from textbox.module.Discriminator.MaliGANDiscriminator import MaliGANDiscriminator


class MaliGAN(GenerativeAdversarialNet):
    r"""MaliGAN is a generative adversarial network using a normalized maximum likelihood optimization.

    """

    def __init__(self, config, dataset):
        super(MaliGAN, self).__init__(config, dataset)
        self.generator = MaliGANGenerator(config, dataset)
        self.discriminator = MaliGANDiscriminator(config, dataset)
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
