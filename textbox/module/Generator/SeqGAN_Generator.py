import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, hidden_size, embedding_size, batch_size, vocabulary_size, dropout_rate, poem_len, start_token, device):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.device = device
        self.poem_len = poem_len
        self.start_token = start_token

        self.LSTM = nn.LSTM(embedding_size, hidden_size)
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.vocab_projection = nn.Linear(hidden_size, vocabulary_size)

    def pre_train(self, datas): # b * len
        datas = datas.permute(1, 0) # len * b
        data_embedding = self.word_embedding(datas) # len * b * e
        output, _ = self.LSTM(data_embedding) # len * b * h
        P = F.log_softmax(self.vocab_projection(output), dim = -1) # len * b * v

        target_word_prob = torch.gather(P[ : -1], index = datas[1 : ].unsqueeze(-1), dim = -1).squeeze(-1) # len * b
        losses = -target_word_prob.mean()
        return losses
    
    def generate(self):
        self.eval()
        with torch.no_grad():
            h_prev = torch.zeros(1, self.batch_size, self.hidden_size, device = self.device) # b * h
            o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device = self.device) # b * h
            X = self.word_embedding(torch.tensor([self.start_token], device = self.device)).expand(self.batch_size, -1).unsqueeze(0) # 1 * b * e

            candidate = torch.zeros(self.poem_len, self.batch_size, dtype = torch.long, device = self.device) # len * b
            for i in range(self.poem_len):
                output, (h_prev, o_prev) = self.LSTM(X, (h_prev, o_prev))
                P = F.softmax(self.vocab_projection(output), dim = -1).squeeze(0) # b * v
                
                for j in range(self.batch_size):
                    candidate[i][j] = torch.multinomial(P[j], 1)[0]
                
                X = self.word_embedding(candidate[i]).unsqueeze(0) # 1 * b * e
        self.train()
        return candidate.permute(1, 0) # b * len