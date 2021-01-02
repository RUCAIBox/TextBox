# @Time   : 2020/11/20
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn

r"""
RankGAN
################################################
Reference:
    Lin et al. "Adversarial Ranking for Language Generation" in NIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.RankGANGenerator import RankGANGenerator
from textbox.module.Discriminator.RankGANDiscriminator import RankGANDiscriminator


class RankGAN(GenerativeAdversarialNet):
    r"""RankGAN is a generative adversarial network consisting of a generator and a ranker.
    The ranker is trained to rank the machine-written sentences lower than human-written sentences with respect to reference sentences.
    The generator is trained to synthesize sentences that can be ranked higher than the human-written one.
    We implement the model following the original author.
    """

    def __init__(self, config, dataset):
        super(RankGAN, self).__init__(config, dataset)
        self.generator = RankGANGenerator(config, dataset)
        self.discriminator = RankGANDiscriminator(config, dataset)
        self.pad_idx = dataset.padding_token_idx
        self.ref_size = config['ref_size']

    def calculate_g_train_loss(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus)
    
    def calculate_d_train_loss(self, real_data, fake_data, ref_data, epoch_idx):
        return self.discriminator.calculate_loss(real_data, fake_data, ref_data)
    
    def calculate_g_adversarial_loss(self, ref_data, epoch_idx):
        self.discriminator.eval()
        loss = self.generator.adversarial_loss(ref_data, self.discriminator.get_rank_scores)
        self.discriminator.train()
        return loss
    
    def calculate_nll_test(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus, nll_test=True)
        
    def generate(self, eval_data):
        return self.generator.generate(eval_data)

    def sample(self, sample_num):
        samples = self.generator.sample(sample_num)
        return samples
