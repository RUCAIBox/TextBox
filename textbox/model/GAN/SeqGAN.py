# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from textbox.utils import InputType
from textbox.model.GAN.abstract_gan import AbstractGAN
# from recbole.model.loss import BPRLoss
from textbox.module.Generator.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization


class SeqGAN(AbstractGAN):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(RNN, self).__init__(config, dataset)
        assert config['rnn_type'] == "lstm"
        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        self.max_length = config['max_length']

        self.module_def(config, dataset)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def module_def(self, config, dataset):
        self.D = Discriminator(config, dataset)
        self.G = RNN(config, dataset)

    def generate(self, eval_data):
        return self.G.generate(eval_data)

    def calculate_loss_G(self, corpus):
        return self.G.calculate_loss(corpus)

    def calculate_loss_D(self, text, label):
        return self.D.calculate_loss(text, label)

    def _init_hidden(self, X):
        return self.G.decoder.init_hidden(X)

    def MCTS(self, corpus):
        input_text = corpus['target_text']
        batch_size = input_text.size(0)

        G = self.G
        D = self.D
        D.eval()
        X = torch.LongTensor([self.sos_token_idx] * batch_size, device=self.device).unsqueeze(0).to(self.device)
        X = self.G.get_embedding(X)
        h_prev, o_prev = self._init_hidden(X)
        rewards = 0
        for t in range(1, self.max_length):
            logits, (h_prev, o_prev) = G(X, (h_prev, o_prev))
            P = F.log_softmax(logits, dim=-1)  # b * v
            word_t = input_text[:, t]  # b
            P_t = torch.gather(P, 1, word_t.unsqueeze(1)).squeeze(1)  # b
            X = G.token_embedder(word_t).unsqueeze(0)  # 1 * b * e

            G.eval()
            with torch.no_grad():
                # MCTS_X = word_t.repeat_interleave(MCTS_num)  # (b * M)
                MCTS_X = torch.repeat_interleave(word_t, MCTS_num)
                MCTS_X = self.G.get_embedding(MCTS_X).unsqueeze(0)
                MCTS_h_prev = h_prev.clone().detach().repeat_interleave(MCTS_num, dim=1)  # 1 * (b * M) * h
                MCTS_o_prev = o_prev.clone().detach().repeat_interleave(MCTS_num, dim=1)  # 1 * (b * M) * h
                MCTS_output = torch.zeros(self.max_length, batch_size * MCTS_num, dtype=torch.long,
                                          device=device)  # len * (b * M)
                for i in range(max_length - t - 1):
                    MCTS_logits, (MCTS_h_prev, MCTS_o_prev) = self.G(MCTS_X, (MCTS_h_prev, MCTS_o_prev))
                    P = F.softmax(MCTS_logits, dim=-1).squeeze(0)  # (b * M) * v
                    for j in range(P.shape[0]):
                        MCTS_output[i + t + 1][j] = torch.multinomial(P[j], 1)[0]
                    MCTS_X = self.G.get_embedding(MCTS_output[i + t + 1]).unsqueeze(0)  # 1 * (b * M) * e

                MCTS_output = MCTS_output.permute(1, 0)  # (b * M) * len
                MCTS_output[:, : t + 1] = input_text[:, : t + 1].repeat_interleave(MCTS_num, dim=0)

                D_out = D.forward(MCTS_output)  # (b * M)
                reward = D_out.reshape(batch_size, MCTS_num).mean(dim=1)  # b

            G.train()
            mask = (word_t != pad_idx) & (word_t != end_idx)
            reward = reward * P_t * mask
            reward = reward[reward.nonzero(as_tuple=True)]
            if (reward.shape[0]):
                rewards += reward.mean()
        D.train()

        return rewards
