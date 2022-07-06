# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

r"""
SeqGAN Generator
#####################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from textbox.model.abstract_generator import UnconditionalGenerator


class SeqGANGenerator(UnconditionalGenerator):
    r"""The generator of SeqGAN.
    """

    def __init__(self, config, dataset):
        super(SeqGANGenerator, self).__init__(config, dataset)
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['generator_embedding_size']
        self.max_length = config['seq_len'] + 2
        self.monte_carlo_num = config['Monte_Carlo_num']

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size)
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        self.vocab_projection = nn.Linear(self.hidden_size, self.vocab_size)

    def calculate_loss(self, corpus, nll_test=False):
        r"""Calculate the generated loss of corpus.

        Args:
            corpus (Corpus): The corpus to be calculated.
            nll_test (Bool): Optional; if nll_test is True the loss is calculated in sentence level rather than in word level.
        
        Returns:
            torch.Tensor: The calculated loss of corpus, shape: [].
        """
        datas = corpus['target_idx']  # b * len
        datas = datas.permute(1, 0)  # len * b
        data_embedding = self.word_embedding(datas[:-1])  # len * b * e
        output, _ = self.LSTM(data_embedding)  # len * b * h
        logits = self.vocab_projection(output)  # len * b * v

        target_word = datas[1:]  # len * b
        target_word_prob = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_word.reshape(-1),
            ignore_index=self.padding_token_idx,
            reduction='none'
        )  # (len * b)
        target_word_prob = target_word_prob.reshape_as(target_word)  # len * b
        if (nll_test):
            loss = target_word_prob.sum(dim=0)
        else:
            length = corpus['target_length'] - 1  # b
            loss = target_word_prob.sum(dim=0) / length.float()  # b
        return loss.mean()

    def _sample_batch(self):
        r"""Sample a batch of generated sentence indice.

        Returns:
            torch.Tensor: The generated sentence indice, shape: [batch_size, max_length].
        """
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            prev_state = (h_prev, o_prev)
            X = self.word_embedding(
                torch.tensor([self.sos_token_idx] * self.batch_size, dtype=torch.long, device=self.device)
            ).unsqueeze(0)  # 1 * b * e
            sentences = torch.zeros((self.max_length, self.batch_size), dtype=torch.long, device=self.device)
            sentences[0] = self.sos_token_idx

            for i in range(1, self.max_length):
                output, prev_state = self.LSTM(X, prev_state)
                P = F.softmax(self.vocab_projection(output), dim=-1).squeeze(0)  # b * v
                for j in range(self.batch_size):
                    sentences[i][j] = torch.multinomial(P[j], 1)[0]
                X = self.word_embedding(sentences[i]).unsqueeze(0)  # 1 * b * e

            sentences = sentences.permute(1, 0)  # b * l

            for i in range(self.batch_size):
                end_pos = (sentences[i] == self.eos_token_idx).nonzero(as_tuple=False)
                if (end_pos.shape[0]):
                    sentences[i][end_pos[0][0] + 1:] = self.padding_token_idx

        self.train()
        return sentences

    def sample(self, sample_num):
        r"""Sample sample_num generated sentence indice.

        Args:
            sample_num (int): The number to generate.

        Returns:
            torch.Tensor: The generated sentence indice, shape: [sample_num, max_length].
        """
        samples = []
        batch_num = math.ceil(sample_num / self.batch_size)
        for _ in range(batch_num):
            samples.append(self._sample_batch())
        samples = torch.cat(samples, dim=0)
        return samples[:sample_num, :]

    def generate(self, batch_data, eval_data):
        r"""Generate tokens of sentences using eval_data.

        Args:
            batch_data (Corpus): Single batch corpus information of evaluation data.
            eval_data : Common information of all evaluation data.

        Returns:
            List[List[str]]: The generated tokens of each sentence.
        """
        generate_corpus = []
        idx2token = eval_data.idx2token
        batch_size = len(batch_data['target_text'])

        for _ in range(batch_size):
            h_prev = torch.zeros(1, 1, self.hidden_size, device=self.device)  # 1 * 1 * h
            o_prev = torch.zeros(1, 1, self.hidden_size, device=self.device)  # 1 * 1 * h
            prev_state = (h_prev, o_prev)
            X = self.word_embedding(
                torch.tensor([[self.sos_token_idx]], dtype=torch.long, device=self.device)
            )  # 1 * 1 * e
            generate_tokens = []

            for _ in range(self.max_length):
                output, prev_state = self.LSTM(X, prev_state)
                P = F.softmax(self.vocab_projection(output), dim=-1).squeeze()  # v
                token = torch.multinomial(P, 1)[0]
                X = self.word_embedding(torch.tensor([[token]], dtype=torch.long, device=self.device))  # 1 * 1 * e
                if (token.item() == self.eos_token_idx):
                    break
                else:
                    generate_tokens.append(idx2token[token.item()])

            generate_corpus.append(generate_tokens)

        return generate_corpus

    def adversarial_loss(self, discriminator_func):
        r"""Calculate the adversarial generator loss guided by discriminator_func.

        Args:
            discriminator_func (function): The function provided from discriminator to calculated the loss of generated sentence.
        
        Returns:
            torch.Tensor: The calculated adversarial loss, shape: [].
        """
        fake_samples = self.sample(self.batch_size)
        h_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
        o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
        X = self.word_embedding(torch.tensor([self.sos_token_idx] * self.batch_size,
                                             device=self.device)).unsqueeze(0)  # 1 * b * e

        rewards = 0
        for t in range(1, self.max_length):
            output, (h_prev, o_prev) = self.LSTM(X, (h_prev, o_prev))
            logits = self.vocab_projection(output).squeeze(0)  # b * v
            P = F.log_softmax(logits, dim=-1)  # b * v
            word_t = fake_samples[:, t]  # b
            P_t = torch.gather(P, 1, word_t.unsqueeze(1)).squeeze(1)  # b
            X = self.word_embedding(word_t).unsqueeze(0)  # 1 * b * e

            self.eval()
            with torch.no_grad():
                monte_carlo_X = word_t.repeat_interleave(self.monte_carlo_num)  # (b * M)
                monte_carlo_X = self.word_embedding(monte_carlo_X).unsqueeze(0)  # 1 * (b * M) * e
                monte_carlo_h_prev = h_prev.clone().detach().repeat_interleave(
                    self.monte_carlo_num, dim=1
                )  # 1 * (b * M) * h
                monte_carlo_o_prev = o_prev.clone().detach().repeat_interleave(
                    self.monte_carlo_num, dim=1
                )  # 1 * (b * M) * h
                monte_carlo_output = torch.zeros(
                    self.max_length, self.batch_size * self.monte_carlo_num, dtype=torch.long, device=self.device
                )  # len * (b * M)

                for i in range(self.max_length - t - 1):
                    output, (monte_carlo_h_prev,
                             monte_carlo_o_prev) = self.LSTM(monte_carlo_X, (monte_carlo_h_prev, monte_carlo_o_prev))
                    P = F.softmax(self.vocab_projection(output), dim=-1).squeeze(0)  # (b * M) * v
                    for j in range(P.shape[0]):
                        monte_carlo_output[i + t + 1][j] = torch.multinomial(P[j], 1)[0]
                    monte_carlo_X = self.word_embedding(monte_carlo_output[i + t + 1]).unsqueeze(0)  # 1 * (b * M) * e

                monte_carlo_output = monte_carlo_output.permute(1, 0)  # (b * M) * len
                monte_carlo_output[:, :t + 1] = fake_samples[:, :t + 1].repeat_interleave(self.monte_carlo_num, dim=0)

                discriminator_out = discriminator_func(monte_carlo_output)  # (b * M)
                reward = discriminator_out.reshape(self.batch_size, self.monte_carlo_num).mean(dim=1)  # b

            self.train()
            mask = word_t != self.padding_token_idx
            reward = reward * P_t * mask.float()
            mask_sum = mask.sum()
            if (mask_sum):
                rewards += reward.sum() / mask_sum

        return -rewards
