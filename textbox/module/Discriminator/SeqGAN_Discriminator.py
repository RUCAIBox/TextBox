import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, batch_size, embedding_size, vocabulary_size, filter_sizes, filter_nums, poem_len, dropout_rate, l2_reg_lambda, device):
        super().__init__()

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.poem_len = poem_len
        self.filter_sum = sum(filter_nums)
        self.l2_reg_lambda = l2_reg_lambda
        self.device = device

        self.word_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.filters = nn.ModuleList([])

        for (filter_size, filter_num) in zip(filter_sizes, filter_nums):
            self.filters.append(
                nn.Sequential(
                nn.Conv2d(1, filter_num, (filter_size, embedding_size)),
                nn.ReLU(),
                nn.MaxPool2d((poem_len - filter_size + 1, 1))))

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
    
    def pre_train(self, data, label):
        y_hat = self.forward(data)

        loss = F.binary_cross_entropy(y_hat, label)
        loss += self.l2_reg_lambda * (self.W_O.weight.norm() + self.W_O.bias.norm())

        return loss