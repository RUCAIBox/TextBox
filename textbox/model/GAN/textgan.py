# @Time   : 2020/11/24
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.TextGANGenerator import TextGANGenerator
from textbox.module.Discriminator.TextGANDiscriminator import TextGANDiscriminator


class TextGAN(GenerativeAdversarialNet):
    """Adversarial Feature Matching for Text Generation

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(TextGAN, self).__init__(config, dataset)
        self.generator = TextGANGenerator(config, dataset)
        self.discriminator = TextGANDiscriminator(config, dataset)
        self.pad_idx = dataset.padding_token_idx

    def calculate_g_train_loss(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus)
    
    def calculate_d_train_loss(self, real_data, fake_data, z, epoch_idx):
        real_data = F.one_hot(real_data, num_classes = self.generator.vocab_size)
        return self.discriminator.calculate_loss(real_data, fake_data, z)
    
    def calculate_g_adversarial_loss(self, real_data, epoch_idx):
        self.discriminator.eval()
        real_data = F.one_hot(real_data, num_classes = self.generator.vocab_size)
        loss = self.generator.adversarial_loss(real_data, self.discriminator.calculate_g_loss)
        self.discriminator.train()
        return loss
    
    def calculate_nll_test(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus, nll_test=True)

    def generate(self, eval_data):
        return self.generator.generate(eval_data)

    def sample(self):
        samples, z = self.generator.sample()
        return samples, z
