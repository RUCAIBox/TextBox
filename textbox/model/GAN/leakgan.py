# @Time   : 2020/11/19
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn


from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textbox.utils import InputType
from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.LeakGANGenerator import LeakGANGenerator
from textbox.module.Discriminator.LeakGANDiscriminator import LeakGANDiscriminator


class LeakGAN(GenerativeAdversarialNet):
    """Sequence Generative Adversarial Nets with Policy Gradient

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(LeakGAN, self).__init__(config, dataset)
        self.generator = LeakGANGenerator(config, dataset)
        self.discriminator = LeakGANDiscriminator(config, dataset)
        self.dis_sample_num = config['d_sample_num']
        self.start_idx = dataset.sos_token_idx
        self.pad_idx = dataset.padding_token_idx

    def calculate_g_train_loss(self, corpus, epoch_idx):
        return self.generator.pretrain_loss(corpus, self.discriminator) # corpus: target_text

    def calculate_d_train_loss(self, real_data, fake_data, epoch_idx):
        return self.discriminator.calculate_loss(real_data[:, 1:], fake_data)

    def calculate_g_adversarial_loss(self, epoch_idx):
        self.discriminator.eval()
        loss = self.generator.adversarial_loss(self.discriminator)
        self.discriminator.train()
        return loss # (manager_loss, worker_loss)

    def generate(self, eval_data):
        return self.generator.generate(eval_data, self.discriminator)

    def sample(self, sample_num):
        samples = self.generator.sample(sample_num, self.discriminator, self.start_idx)
        return samples

    def calculate_nll_test(self, corpus, epoch_idx):
        return self.generator.calculate_loss(corpus, self.discriminator)