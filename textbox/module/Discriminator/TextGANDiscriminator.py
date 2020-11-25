# @Time   : 2020/11/24
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
from textbox.model.abstract_generator import UnconditionalGenerator


class TextGANDiscriminator(UnconditionalGenerator):
    def __init__(self, config, dataset):
        super(TextGANDiscriminator, self).__init__(config, dataset)

        self.embedding_size = config['discriminator_embedding_size']
        self.hidden_size = config['hidden_size']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.mmd_lambda = config['mmd_lambda']
        self.recon_lambda = config['recon_lambda']
        self.dropout_rate = config['dropout_rate']
        self.filter_sizes = config['filter_sizes']
        self.filter_nums = config['filter_nums']
        self.max_length = config['max_seq_length'] + 2
        self.gaussian_sigmas = torch.tensor(config['gaussian_sigmas'], device=self.device)
        self.pad_idx = dataset.padding_token_idx
        self.vocab_size = dataset.vocab_size
        self.filter_sum = sum(self.filter_nums)

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx = self.pad_idx)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.filters = nn.ModuleList([])

        for (filter_size, filter_num) in zip(self.filter_sizes, self.filter_nums):
            self.filters.append(
                nn.Sequential(
                nn.Conv2d(1, filter_num, (filter_size, self.embedding_size)),
                nn.ReLU(),
                nn.MaxPool2d((self.max_length - filter_size + 1, 1))))

        self.W_O = nn.Linear(self.filter_sum, 1)
        self.recon = nn.Linear(self.filter_sum, self.hidden_size)
    
    def feature(self, data): # b * len * v
        data = torch.matmul(data.float(), self.word_embedding.weight).unsqueeze(1) # b * len * e -> b * 1 * len * e
        combined_outputs = []
        for CNN_filter in self.filters:
            output = CNN_filter(data).squeeze(-1).squeeze(-1) # b * f_n * 1 * 1 -> b * f_n
            combined_outputs.append(output)
        combined_outputs = torch.cat(combined_outputs, 1) # b * tot_f_n
        combined_outputs = self.dropout(combined_outputs)

        return combined_outputs

    def forward(self, data): # b * len * v
        features = self.feature(data) # b * tot_f_n
        y_hat = torch.sigmoid(self.W_O(features)).squeeze(1) # b
        return y_hat
    
    def calculate_gan_loss(self, real_data, fake_data):
        real_y = self.forward(real_data)
        fake_y = self.forward(fake_data)
        real_label = torch.ones_like(real_y)
        fake_label = torch.zeros_like(fake_y)

        real_loss = F.binary_cross_entropy(real_y, real_label)
        fake_loss = F.binary_cross_entropy(fake_y, fake_label)
        loss = (real_loss + fake_loss) / 2

        return loss

    def compute_pairwise_distances(self, x, y): # b * tot_f_n, b * tot_f_n
        return torch.pow((x.unsqueeze(2) - y.T).norm(dim = 1), 2).T # b * t * 1 - t * b -> b * t * b - b * t * b -> b * t * b -> b * b

    def gaussian_kernel_matrix(self, x, y): # b * tot_f_n, b * tot_f_n
        beta = 1. / (2. * self.gaussian_sigmas.unsqueeze(1)) # sig_n * 1
        dist = self.compute_pairwise_distances(x, y) # b * b
        s = torch.matmul(beta, dist.reshape(1, -1)) # sig_n * 1 x 1 * (b * b) -> sig_n * (b * b)
        return torch.exp(-s).sum(dim = 0).reshape_as(dist) # sig_n * (b * b) -> (b * b) -> b * b

    def calculate_mmd_loss(self, x, y):
        cost = self.gaussian_kernel_matrix(x, x).mean()
        cost += self.gaussian_kernel_matrix(y, y).mean()
        cost -= 2 * self.gaussian_kernel_matrix(x, y).mean()
        return cost

    def calculate_recon_loss(self, fake_feature, z): # b * tot_f_n, b * h
        z_hat = self.recon(fake_feature) # b * h
        return (z - z_hat).norm(dim = 1).mean() # b * h -> b -> 1

    def calculate_g_loss(self, real_data, fake_data):
        real_feature = self.feature(real_data) # b * tot_f_n
        fake_feature = self.feature(fake_data) # b * tot_f_n
        mmd_loss = self.mmd_lambda * self.calculate_mmd_loss(real_feature, fake_feature)
        return mmd_loss

    def calculate_loss(self, real_data, fake_data, z):
        gan_loss = self.calculate_gan_loss(real_data, fake_data)
        real_feature = self.feature(real_data) # b * tot_f_n
        fake_feature = self.feature(fake_data) # b * tot_f_n
        mmd_loss = -self.mmd_lambda * self.calculate_mmd_loss(real_feature, fake_feature)
        recon_loss = self.recon_lambda * self.calculate_recon_loss(fake_feature, z)
        l2_reg_loss = self.l2_reg_lambda * (self.W_O.weight.norm() + self.W_O.bias.norm())

        loss = gan_loss + mmd_loss + recon_loss + l2_reg_loss
        return loss