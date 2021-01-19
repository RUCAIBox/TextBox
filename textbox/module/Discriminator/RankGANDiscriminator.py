# @Time   : 2020/11/20
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn

r"""
RankGAN Discriminator
#####################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from textbox.model.abstract_generator import UnconditionalGenerator


class RankGANDiscriminator(UnconditionalGenerator):
    r"""RankGANDiscriminator is a ranker which can endow a relative rank among the sequences when given a reference.
    The ranker is designed with the convolutional neural network.
    """

    def __init__(self, config, dataset):
        super(RankGANDiscriminator, self).__init__(config, dataset)

        self.embedding_size = config['discriminator_embedding_size']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.dropout_rate = config['dropout_rate']
        self.filter_sizes = config['filter_sizes']
        self.filter_nums = config['filter_nums']
        self.max_length = config['max_seq_length'] + 2
        self.pad_idx = dataset.padding_token_idx
        self.vocab_size = dataset.vocab_size
        self.filter_sum = sum(self.filter_nums)
        self.gamma = config['gamma']  # temprature control parameters

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.pad_idx)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.filters = nn.ModuleList([])

        for (filter_size, filter_num) in zip(self.filter_sizes, self.filter_nums):
            self.filters.append(
                nn.Sequential(
                    nn.Conv2d(1, filter_num, (filter_size, self.embedding_size)), nn.ReLU(),
                    nn.MaxPool2d((self.max_length - filter_size + 1, 1))
                )
            )

        self.W_T = nn.Linear(self.filter_sum, self.filter_sum)
        self.W_H = nn.Linear(self.filter_sum, self.filter_sum, bias=False)

    def highway(self, data):
        r"""Apply the highway net to data.

        Args:
            data (torch.Tensor): The original data, shape: [batch_size, total_filter_num].

        Returns:
            torch.Tensor: The data processed after highway net, shape: [batch_size, total_filter_num].
        """
        tau = torch.sigmoid(self.W_T(data))
        non_linear = F.relu(self.W_H(data))
        return self.dropout(tau * non_linear + (1 - tau) * data)

    def forward(self, data):
        r"""Maps concatenated sequence matrices into the embedded feature vectors.

        Args:
            data (torch.Tensor): The sentence data, shape: [batch_size, max_seq_len].

        Returns:
            torch.Tensor: The embedded feature vectors, shape: [batch_size, total_filter_num].
        """
        data = self.word_embedding(data).unsqueeze(1)  # b * len * e -> b * 1 * len * e
        combined_outputs = []
        for CNN_filter in self.filters:
            output = CNN_filter(data).squeeze(-1).squeeze(-1)  # b * f_n * 1 * 1 -> b * f_n
            combined_outputs.append(output)
        combined_outputs = torch.cat(combined_outputs, 1)  # b * tot_f_n

        feature = self.highway(combined_outputs)  # b * tot_f_n

        return feature

    def get_rank_scores(self, sample_data, ref_data):
        r"""Get the ranking score (before softmax) for sample s given reference u.
        
        .. math::
            \alpha(s|u) = cosine(y_s,y_u) = \frac{y_s \cdot y_u}{\parallel y_s \parallel \parallel y_u \parallel}

        Args:
            sample_data (torch.Tensor): The realistic or generated sentence data, shape: [sample_size, max_seq_len].
            ref_data (torch.Tensor): The reference sentence data, shape: [ref_size, max_seq_len].

        Returns:
            torch.Tensor: The ranking score of sample data, shape: [batch_size].
        """
        feature = self.forward(sample_data)  # sample_size * tot_f_n
        ref_feature = self.forward(ref_data)  # ref_size * tot_f_n

        scores = torch.matmul(F.normalize(feature), F.normalize(ref_feature).permute(1, 0))  # sample_size * ref_size
        scores = self.gamma * torch.reshape(torch.sum(scores, 1), [-1])  # sample_size * ref_size -> sample_size

        return scores

    def calculate_loss(self, real_data, fake_data, ref_data):
        r"""Calculate the loss for real data and fake data.
        To rank the human_written sentences higher than the machine-written sentences.

        Args:
            real_data (torch.Tensor): The realistic sentence data, shape: [batch_size, max_seq_len].
            fake_data (torch.Tensor): The generated sentence data, shape: [batch_size, max_seq_len].
            ref_data (torch.Tensor): The reference sentence data, shape: [ref_size, max_seq_len].

        Returns:
            torch.Tensor: The calculated loss of real data and fake data, shape: [].
        """
        # ranking
        sample_data = torch.cat((real_data, fake_data), dim=0)  # 2b * l
        scores = self.get_rank_scores(sample_data, ref_data)  # 2b
        #rank_score = torch.reshape(F.softmax(scores, dim = -1), [-1])
        rank_score = F.softmax(scores, dim=-1)
        log_rank = torch.log(rank_score)  # 2b

        # ranking loss
        real_label = torch.tensor([[0., 1.] for _ in range(real_data.shape[0])], device=self.device)  # b * 2
        fake_label = torch.tensor([[1., 0.] for _ in range(fake_data.shape[0])], device=self.device)  # b * 2
        label = torch.cat((real_label, fake_label), dim=0)  # 2b * 2
        trans_label = label.permute(1, 0)
        pos_ind = trans_label[1]
        neg_ind = trans_label[0]
        pos_loss = torch.sum(pos_ind * log_rank) / torch.sum(pos_ind)
        neg_loss = torch.sum(neg_ind * log_rank) / torch.sum(neg_ind)
        loss = -(pos_loss - neg_loss)

        return loss
