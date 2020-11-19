# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F


from textbox.model.abstract_generator import UnconditionalGenerator


class SeqGANDiscriminator(UnconditionalGenerator):
    def __init__(self, config, dataset):
        super(SeqGANDiscriminator, self).__init__(config, dataset)

        self.embedding_size = config['discriminator_embedding_size']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.dropout_rate = config['dropout_rate']
        self.filter_sizes = config['filter_sizes']
        self.filter_nums = config['filter_nums']
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

        self.W_T = nn.Linear(self.filter_sum, self.filter_sum)
        self.W_H = nn.Linear(self.filter_sum, self.filter_sum, bias = False)
        self.W_O = nn.Linear(self.filter_sum, 1)
    
    def highway(self, data):
        tau = torch.sigmoid(self.W_T(data))
        non_linear = F.relu(self.W_H(data))
        return self.dropout(tau * non_linear + (1 - tau) * data)

    def forward(self, data): # b * len
        data = self.word_embedding(data).unsqueeze(1) # b * len * e -> b * 1 * len * e
        combined_outputs = []
        for CNN_filter in self.filters:
            output = CNN_filter(data).squeeze(-1).squeeze(-1) # b * f_n * 1 * 1 -> b * f_n
            combined_outputs.append(output)
        combined_outputs = torch.cat(combined_outputs, 1) # b * tot_f_n

        C_tilde = self.highway(combined_outputs) # b * tot_f_n
        y_hat = torch.sigmoid(self.W_O(C_tilde)).squeeze(1) # b

        return y_hat

    def calculate_loss(self, real_data, fake_data):
        real_y = self.forward(real_data)
        fake_y = self.forward(fake_data)
        real_label = torch.ones_like(real_y)
        fake_label = torch.zeros_like(fake_y)

        real_loss = F.binary_cross_entropy(real_y, real_label)
        fake_loss = F.binary_cross_entropy(fake_y, fake_label)
        loss = (real_loss + fake_loss) / 2 +  self.l2_reg_lambda * (self.W_O.weight.norm() + self.W_O.bias.norm())

        return loss
