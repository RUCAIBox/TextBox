# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def calculate_g_train_loss(self, corpus):
        return self.generator.calculate_loss(corpus)
    
    def calculate_d_train_loss(self, real_data, fake_data):
        return self.discriminator.calculate_loss(real_data, fake_data)
    
    def calculate_g_adversarial_loss(self, epoch_idx):
        fake_samples = self.generator.sample(self.batch_size)

        D.eval()
        h_prev = torch.zeros(1, self.batch_size, hidden_size, device = device) # 1 * b * h
        o_prev = torch.zeros(1, batch_size, hidden_size, device = device) # 1 * b * h
        X = G.word_embedding(torch.tensor([start_idx] * batch_size, device = device)).unsqueeze(0) # 1 * b * e

        rewards = 0
        for t in range(1, max_length):
            output, (h_prev, o_prev) = G.LSTM(X, (h_prev, o_prev))
            logits = G.vocab_projection(output).squeeze(0) # b * v
            P = F.log_softmax(logits, dim = -1) # b * v
            word_t = samples[ : , t] # b
            P_t = torch.gather(P, 1, word_t.unsqueeze(1)).squeeze(1) # b
            X = G.word_embedding(word_t).unsqueeze(0) # 1 * b * e

            G.eval()
            with torch.no_grad():
                MCTS_X = word_t.repeat_interleave(MCTS_num) # (b * M)
                MCTS_X = G.word_embedding(MCTS_X).unsqueeze(0) # 1 * (b * M) * e
                MCTS_h_prev = h_prev.clone().detach().repeat_interleave(MCTS_num, dim = 1) # 1 * (b * M) * h
                MCTS_o_prev = o_prev.clone().detach().repeat_interleave(MCTS_num, dim = 1) # 1 * (b * M) * h
                MCTS_output = torch.zeros(max_length, batch_size * MCTS_num, dtype = torch.long, device = device) # len * (b * M)

                for i in range(max_length - t - 1):
                    output, (MCTS_h_prev, MCTS_o_prev) = G.LSTM(MCTS_X, (MCTS_h_prev, MCTS_o_prev))
                    P = F.softmax(G.vocab_projection(output), dim = -1).squeeze(0) # (b * M) * v
                    for j in range(P.shape[0]):
                        MCTS_output[i + t + 1][j] = torch.multinomial(P[j], 1)[0]
                    MCTS_X = G.word_embedding(MCTS_output[i + t + 1]).unsqueeze(0) # 1 * (b * M) * e

                MCTS_output = MCTS_output.permute(1, 0) # (b * M) * len
                MCTS_output[ : , : t + 1] = samples[ : , : t + 1].repeat_interleave(MCTS_num, dim = 0)
    
                D_out = D.forward(MCTS_output) # (b * M)
                reward = D_out.reshape(batch_size, MCTS_num).mean(dim = 1) # b
            
            G.train()
            mask = (word_t != pad_idx) & (word_t != end_idx)
            reward = reward * P_t * mask
            reward = reward[reward.nonzero(as_tuple = True)]
            if (reward.shape[0]):
                rewards += reward.mean()
        D.train()
        
        return rewards
