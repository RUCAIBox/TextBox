# @Time   : 2020/11/19
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn


from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Generator.LeakGANGenerator import LeakGANGenerator
from textbox.module.Discriminator.LeakGANDiscriminator import LeakGANDiscriminator


class LeakGAN(GenerativeAdversarialNet):

    def __init__(self, config, dataset):
        super(LeakGAN, self).__init__(config, dataset)
        self.generator = LeakGANGenerator(config, dataset)
        self.discriminator = LeakGANDiscriminator(config, dataset)
        self.dis_sample_num = config['d_sample_num']
        self.start_idx = dataset.sos_token_idx
        self.pad_idx = dataset.padding_token_idx
        self.end_idx = dataset.eos_token_idx
        self.max_length = config['max_seq_length'] + 2

    def calculate_g_train_loss(self, corpus, epoch_idx):
        self.discriminator.eval()
        loss = self.generator.pretrain_loss(corpus, self.discriminator)  # corpus: target_text
        self.discriminator.train()
        return loss

    def calculate_d_train_loss(self, real_data, fake_data, epoch_idx):
        return self.discriminator.calculate_loss(real_data[:, 1:], fake_data)

    def calculate_g_adversarial_loss(self, epoch_idx):
        self.discriminator.eval()
        loss = self.generator.adversarial_loss(self.discriminator)
        self.discriminator.train()
        return loss  # (manager_loss, worker_loss)

    def generate(self, eval_data):
        return self.generator.generate(eval_data, self.discriminator)

    def sample(self, sample_num):
        self.discriminator.eval()
        samples = self.generator.sample(sample_num, self.discriminator, self.start_idx)
        self.discriminator.train()
        return samples

    def calculate_nll_test(self, corpus, epoch_idx):
        targets = self._get_real_data_for_nll_test(corpus)
        targets = targets[:, 1:]
        return self.generator.calculate_loss(targets, self.discriminator)

    def _add_eos(self, data, length):
        batch_size = data.shape[0]
        padded_data = torch.full((batch_size, self.max_length), self.end_idx, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            len = length[i].cpu().data
            padded_data[i, :len] = data[i, :len]
        return padded_data

    def _get_real_data_for_nll_test(self, train_data):
        # specified for nll test and use eos_idx pad not pad_idx to pad
        real_data = train_data['target_idx']
        length = train_data['target_length']
        real_data = self._add_eos(real_data, length)  # bs * seq_len
        return real_data
