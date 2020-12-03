# @Time   : 2020/11/20
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.RankGANGenerator import RankGANGenerator
from textbox.module.Discriminator.RankGANDiscriminator import RankGANDiscriminator


class RankGAN(GenerativeAdversarialNet):
    """Adversarial Ranking for Language Generation

    """
    input_type = InputType.NOISE

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
    
    def generate(self, eval_data):
        return self.generator.generate(eval_data)

    def sample(self, sample_num):
        samples = self.generator.sample(sample_num)
        return samples
