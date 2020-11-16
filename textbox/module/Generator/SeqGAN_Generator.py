import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, hidden_size, embedding_size, batch_size, vocabulary_size, start_idx, end_idx, pad_idx, device):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.device = device
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

        self.LSTM = nn.LSTM(embedding_size, hidden_size)
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=pad_idx)
        self.vocab_projection = nn.Linear(hidden_size, vocabulary_size)

    def pre_train(self, datas):  # b * len
        datas = datas.permute(1, 0)  # len * b

        data_embedding = self.word_embedding(datas[: -1])  # len * b * e
        output, _ = self.LSTM(data_embedding)  # len * b * h
        logits = self.vocab_projection(output)  # len * b * v

        logits = logits.reshape(-1, self.vocabulary_size)  # (len * b) * v
        target = datas[1:].reshape(-1)  # (len * b)

        losses = F.cross_entropy(logits, target, ignore_index=self.pad_idx)
        return losses

    def generate(self, max_length):
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            prev_state = (h_prev, o_prev)
            X = self.word_embedding(
                torch.tensor([self.start_idx] * self.batch_size, dtype=torch.long, device=self.device)).unsqueeze(
                0)  # 1 * b * e
            sentences = torch.zeros((max_length, self.batch_size), dtype=torch.long, device=self.device)
            sentences[0] = self.start_idx

            for i in range(1, max_length):
                output, prev_state = self.LSTM(X, prev_state)
                P = F.softmax(self.vocab_projection(output), dim=-1).squeeze(0)  # b * v
                for j in range(self.batch_size):
                    sentences[i][j] = torch.multinomial(P[j], 1)[0]
                X = self.word_embedding(sentences[i]).unsqueeze(0)  # 1 * b * e

            sentences = sentences.permute(1, 0)  # b * l

            for i in range(self.batch_size):
                end_pos = (sentences[i] == self.end_idx).nonzero()
                if (end_pos.shape[0]):
                    sentences[i][end_pos[0][0] + 1:] = self.pad_idx

        self.train()
        return sentences