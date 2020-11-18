# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textbox.utils import InputType
from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.SeqGANGenerator import SeqGANGenerator
from textbox.module.Discriminator.SeqGANDiscriminator import SeqGANDiscriminator


class SeqGAN(GenerativeAdversarialNet):
    """Sequence Generative Adversarial Nets with Policy Gradient

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(SeqGAN, self).__init__(config, dataset)
        self.generator = SeqGANGenerator(config, dataset)
        self.discriminator = SeqGANDiscriminator(config, dataset)

    def calculate_g_train_loss(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus)
    
    def calculate_d_train_loss(self, real_data, fake_data, epoch_idx):
        return self.discriminator.calculate_loss(real_data, fake_data)
    
    def calculate_g_adversarial_loss(self, epoch_idx):
        self.discriminator.eval()
        loss = self.generator.adversarial_loss(self.discriminator.forward)
        self.discriminator.train()
        return loss
    
    def generate(self, eval_data):
        return self.generator.generate(eval_data)

    def sample(self, sample_num):
        samples = self.generator.sample(sample_num)
        samples_dataloader = DataLoader(samples, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return samples_dataloader