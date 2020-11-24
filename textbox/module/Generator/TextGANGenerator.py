# @Time   : 2020/11/24
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
from textbox.model.abstract_generator import UnconditionalGenerator


class TextGANGenerator(UnconditionalGenerator):
    def __init__(self, config, dataset):
        super(TextGANGenerator, self).__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.embedding_size = config['generator_embedding_size']
        self.max_length = config['max_seq_length'] + 2
        self.monte_carlo_num = config['Monte_Carlo_num']
        self.start_idx = dataset.sos_token_idx
        self.end_idx = dataset.eos_token_idx
        self.pad_idx = dataset.padding_token_idx
        self.vocab_size = dataset.vocab_size

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size)
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx = self.pad_idx)
        self.vocab_projection = nn.Linear(self.hidden_size, self.vocab_size)

    def calculate_loss(self, corpus):
        datas = corpus['target_idx'] # b * len
        datas = datas.permute(1, 0) # len * b
        data_embedding = self.word_embedding(datas[ : -1]) # len * b * e
        output, _ = self.LSTM(data_embedding) # len * b * h
        logits = self.vocab_projection(output) # len * b * v
        
        logits = logits.reshape(-1, self.vocab_size) # (len * b) * v
        target = datas[1 : ].reshape(-1) # (len * b)
        
        losses = F.cross_entropy(logits, target, ignore_index = self.pad_idx)
        return losses
    
    def sample(self):
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.rand(1, self.batch_size, self.hidden_size, device = self.device) # 1 * b * h
            o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device = self.device) # 1 * b * h
            prev_state = (h_prev, o_prev)
            X = self.word_embedding(torch.tensor([self.start_idx] * self.batch_size, dtype = torch.long, device = self.device)).unsqueeze(0) # 1 * b * e
            
            sentences = torch.zeros((self.max_length, self.batch_size), dtype = torch.long, device = self.device) # l * b
            sentences[0] = self.start_idx
            sentences_prob = torch.zeros((self.max_length, self.batch_size, self.vocab_size), device = self.device) # l * b * v
            sentences_prob[0] = F.one_hot(torch.tensor(self.start_idx), num_classes = self.vocab_size)

            for i in range(1, self.max_length):
                output, prev_state = self.LSTM(X, prev_state)
                P = F.softmax(self.vocab_projection(output), dim = -1).squeeze(0) # b * v
                sentences_prob[i] = P
                for j in range(self.batch_size):
                    sentences[i][j] = torch.multinomial(P[j], 1)[0]
                X = self.word_embedding(sentences[i]).unsqueeze(0) # 1 * b * e
            
            sentences = sentences.permute(1, 0) # b * l
            sentences_prob = sentences_prob.permute(1, 0, 2) # b * l * v

            for i in range(self.batch_size):
                end_pos = (sentences[i] == self.end_idx).nonzero()
                if (end_pos.shape[0]):
                    sentences_prob[i][end_pos[0][0] + 1 : ] = F.one_hot(torch.tensor(self.pad_idx), num_classes = self.vocab_size)

        self.train()
        return sentences_prob, h_prev.squeeze(0)

    def generate(self, eval_data):
        self.eval()
        generate_corpus = []
        number_to_gen = 10
        idx2token = eval_data.idx2token

        with torch.no_grad():
            for _ in range(number_to_gen):
                h_prev = torch.zeros(1, 1, self.hidden_size, device = self.device) # 1 * 1 * h
                o_prev = torch.zeros(1, 1, self.hidden_size, device = self.device) # 1 * 1 * h
                prev_state = (h_prev, o_prev)
                X = self.word_embedding(torch.tensor([[self.start_idx]], dtype = torch.long, device = self.device)) # 1 * 1 * e
                generate_tokens = []

                for _ in range(self.max_length):
                    output, prev_state = self.LSTM(X, prev_state)
                    P = F.softmax(self.vocab_projection(output), dim = -1).squeeze() # v
                    token = torch.multinomial(P, 1)[0]
                    X = self.word_embedding(torch.tensor([[token]], dtype = torch.long, device = self.device)) # 1 * 1 * e
                    if (token.item() == self.end_idx):
                        break
                    else:
                        generate_tokens.append(idx2token[token.item()])
                
                generate_corpus.append(generate_tokens)
                
        self.train()
        return generate_corpus
    
    def adversarial_loss(self, discriminator_func):
        fake_samples = self.sample(self.batch_size) # b * l * v
        y = discriminator_func(fake_samples) # b
        label = torch.ones_like(y)
        loss = F.binary_cross_entropy(y, label)
        return loss
