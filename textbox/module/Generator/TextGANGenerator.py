# @Time   : 2020/11/24
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

r"""
TextGAN Generator
#####################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from textbox.model.abstract_generator import UnconditionalGenerator


class TextGANGenerator(UnconditionalGenerator):
    r"""The generator of TextGAN.
    """

    def __init__(self, config, dataset):
        super(TextGANGenerator, self).__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.embedding_size = config['generator_embedding_size']
        self.max_length = config['max_seq_length'] + 2
        self.eval_generate_num = config['eval_generate_num']
        self.start_idx = dataset.sos_token_idx
        self.end_idx = dataset.eos_token_idx
        self.pad_idx = dataset.padding_token_idx
        self.vocab_size = dataset.vocab_size

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size)
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.pad_idx)
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
        h_prev = torch.randn(1, datas.size(1), self.hidden_size, device=self.device)  # 1 * b * h
        o_prev = torch.zeros(1, datas.size(1), self.hidden_size, device=self.device)  # 1 * b * h
        prev_state = (h_prev, o_prev)
        output, _ = self.LSTM(data_embedding, prev_state)  # len * b * h
        logits = self.vocab_projection(output)  # len * b * v

        target_word = datas[1:]  # len * b
        target_word_prob = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target_word.reshape(-1), ignore_index=self.pad_idx, reduction='none'
        )  # (len * b)
        target_word_prob = target_word_prob.reshape_as(target_word)  # len * b
        if (nll_test):
            loss = target_word_prob.sum(dim=0)
        else:
            length = corpus['target_length'] - 1  # b
            loss = target_word_prob.sum(dim=0) / length.float()  # b
        return loss.mean()

    def sample(self):
        r"""Sample a batch of generated sentence indice.

        Returns:
            torch.Tensor: The generated sentence indice, shape: [batch_size, max_seq_length].
            torch.Tensor: The latent code of the generated sentence, shape: [batch_size, hidden_size].
        """
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.randn(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            prev_state = (h_prev, o_prev)
            X = self.word_embedding(
                torch.tensor([self.start_idx] * self.batch_size, dtype=torch.long, device=self.device)
            ).unsqueeze(0)  # 1 * b * e

            sentences = torch.zeros((self.max_length, self.batch_size), dtype=torch.long, device=self.device)  # l * b
            sentences[0] = self.start_idx
            sentences_prob = torch.zeros((self.max_length, self.batch_size, self.vocab_size),
                                         device=self.device)  # l * b * v
            sentences_prob[0] = F.one_hot(torch.tensor(self.start_idx), num_classes=self.vocab_size)

            for i in range(1, self.max_length):
                output, prev_state = self.LSTM(X, prev_state)
                P = F.softmax(self.vocab_projection(output), dim=-1).squeeze(0)  # b * v
                sentences_prob[i] = P
                for j in range(self.batch_size):
                    sentences[i][j] = torch.multinomial(P[j], 1)[0]
                X = self.word_embedding(sentences[i]).unsqueeze(0)  # 1 * b * e

            sentences = sentences.permute(1, 0)  # b * l
            sentences_prob = sentences_prob.permute(1, 0, 2)  # b * l * v

            for i in range(self.batch_size):
                end_pos = (sentences[i] == self.end_idx).nonzero(as_tuple=False)
                if (end_pos.shape[0]):
                    sentences_prob[i][end_pos[0][0] +
                                      1:] = F.one_hot(torch.tensor(self.pad_idx), num_classes=self.vocab_size)

        self.train()
        return sentences_prob, h_prev.squeeze(0)

    def generate(self, eval_data):
        r"""Generate tokens of sentences using eval_data.

        Args:
            eval_data (Corpus): The corpus information of evaluation data.

        Returns:
            List[List[str]]: The generated tokens of each sentence.
        """
        self.eval()
        generate_corpus = []
        idx2token = eval_data.idx2token

        with torch.no_grad():
            for _ in range(self.eval_generate_num):
                h_prev = torch.zeros(1, 1, self.hidden_size, device=self.device)  # 1 * 1 * h
                o_prev = torch.zeros(1, 1, self.hidden_size, device=self.device)  # 1 * 1 * h
                prev_state = (h_prev, o_prev)
                X = self.word_embedding(
                    torch.tensor([[self.start_idx]], dtype=torch.long, device=self.device)
                )  # 1 * 1 * e
                generate_tokens = []

                for _ in range(self.max_length):
                    output, prev_state = self.LSTM(X, prev_state)
                    P = F.softmax(self.vocab_projection(output), dim=-1).squeeze()  # v
                    token = torch.multinomial(P, 1)[0]
                    X = self.word_embedding(torch.tensor([[token]], dtype=torch.long, device=self.device))  # 1 * 1 * e
                    if (token.item() == self.end_idx):
                        break
                    else:
                        generate_tokens.append(idx2token[token.item()])

                generate_corpus.append(generate_tokens)

        self.train()
        return generate_corpus

    def adversarial_loss(self, real_data, discriminator_func):
        r"""Calculate the adversarial generator loss of real_data guided by discriminator_func.

        Args:
            real_data (torch.Tensor): The realistic sentence data, shape: [batch_size, max_seq_len].
            discriminator_func (function): The function provided from discriminator to calculated the loss of generated sentence.
        
        Returns:
            torch.Tensor: The calculated adversarial loss, shape: [].
        """
        fake_samples, _ = self.sample()
        loss = discriminator_func(real_data, fake_samples)
        return loss
