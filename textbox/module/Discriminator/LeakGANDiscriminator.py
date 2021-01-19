# @Time   : 2020/11/19
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn

r"""
LeakGAN Discriminator
#####################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from textbox.model.abstract_generator import UnconditionalGenerator


class LeakGANDiscriminator(UnconditionalGenerator):
    r"""CNN based discriminator for leakgan extracting feature of current sentence
    """

    def __init__(self, config, dataset):
        super(LeakGANDiscriminator, self).__init__(config, dataset)

        self.embedding_size = config['discriminator_embedding_size']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.dropout_rate = config['dropout_rate']
        self.filter_sizes = config['filter_sizes']
        self.filter_nums = config['filter_nums']
        self.max_length = config['max_seq_length'] + 1
        self.pad_idx = dataset.padding_token_idx
        self.vocab_size = dataset.vocab_size
        self.filter_sum = sum(self.filter_nums)

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.filters = nn.ModuleList([])

        for (filter_size, filter_num) in zip(self.filter_sizes, self.filter_nums):
            self.filters.append(
                nn.Sequential(
                    nn.Conv2d(1, filter_num, (filter_size, self.embedding_size), stride=1, padding=0, bias=True),
                    nn.ReLU(), nn.MaxPool2d((self.max_length - filter_size + 1, 1), stride=1, padding=0)
                )
            )

        self.W_T = nn.Linear(self.filter_sum, self.filter_sum)
        self.W_H = nn.Linear(self.filter_sum, self.filter_sum, bias=False)
        self.W_O = nn.Linear(self.filter_sum, 2)

    def highway(self, data):
        tau = torch.sigmoid(self.W_T(data))
        non_linear = F.relu(self.W_H(data))

        return self.dropout(tau * non_linear + (1 - tau) * data)

    def forward(self, data):  # b * len
        r"""Get current sentence feature by CNN
        """
        C_tilde = self.get_feature(data)
        pred = self.W_O(C_tilde)

        return pred

    def get_feature(self, inp):
        r"""Get feature vector of given sentences

        Args:
            inp: batch_size * max_seq_len

        Returns:
            batch_size * feature_dim
        """
        data = self.word_embedding(inp).unsqueeze(1)  # b * len * e -> b * 1 * len * e
        combined_outputs = []
        for CNN_filter in self.filters:
            output = CNN_filter(data)
            combined_outputs.append(output)
        combined_outputs = torch.cat(combined_outputs, 1)  # b * tot_f_n :pred
        combined_outputs = combined_outputs.squeeze(dim=3).squeeze(dim=2)

        C_tilde = self.highway(combined_outputs)  # b * tot_f_n

        return C_tilde

    def calculate_loss(self, real_data, fake_data):
        r"""Calculate discriminator loss and acc
        """
        real_y = self.forward(real_data)
        fake_y = self.forward(fake_data)
        pre_logits = torch.cat([real_y, fake_y], dim=0)

        real_label = torch.ones_like(real_y, dtype=torch.int64)[:, 0].long()  # [1,1,1]
        fake_label = torch.zeros_like(fake_y, dtype=torch.int64)[:, 0].long()  # [0,0,0]
        label = torch.cat([real_label, fake_label], dim=-1)
        loss = F.cross_entropy(pre_logits, label)

        loss = loss + self.l2_reg_lambda * (torch.norm(self.W_O.weight, 2) + torch.norm(self.W_O.bias, 2))

        pred = torch.cat([real_y, fake_y], dim=0)  # bs*2
        target = torch.cat([real_label, fake_label], dim=0)  # bs
        acc = torch.sum((pred.argmax(dim=-1) == target)).item()
        acc = acc / pred.size()[0]
        return loss, acc
