# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F


from textbox.model.abstract_generator import UnconditionalGenerator

class SeqGANGenerator(UnconditionalGenerator):
    def __init__(self, config, dataset):
        super(SeqGANGenerator, self).__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.embedding_size = config['generator_embedding_size']
        self.max_length = config['max_seq_length']
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
    
    def sample_batch(self):
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.zeros(1, self.batch_size, self.hidden_size, device = self.device) # 1 * b * h
            o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device = self.device) # 1 * b * h
            prev_state = (h_prev, o_prev)
            X = self.word_embedding(torch.tensor([self.start_idx] * self.batch_size, dtype = torch.long, device = self.device)).unsqueeze(0) # 1 * b * e
            sentences = torch.zeros((self.max_length, self.batch_size), dtype = torch.long, device = self.device)
            sentences[0] = self.start_idx

            for i in range(1, self.max_length):
                output, prev_state = self.LSTM(X, prev_state)
                P = F.softmax(self.vocab_projection(output), dim = -1).squeeze(0) # b * v
                for j in range(self.batch_size):
                    sentences[i][j] = torch.multinomial(P[j], 1)[0]
                X = self.word_embedding(sentences[i]).unsqueeze(0) # 1 * b * e
            
            sentences = sentences.permute(1, 0) # b * l

            for i in range(self.batch_size):
                end_pos = (sentences[i] == self.end_idx).nonzero()
                if (end_pos.shape[0]):
                    sentences[i][end_pos[0][0] + 1 : ] = self.pad_idx

        self.train()
        return sentences

    def sample(self, sample_num):
        samples = []
        batch_num = sample_num // self.batch_size
        for _ in range(batch_num):
            samples.append(self.sample_batch())
        samples = torch.cat(samples, dim = 0)
        return samples

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
        fake_samples = self.sample(self.batch_size)
        h_prev = torch.zeros(1, self.batch_size, self.hidden_size, device = self.device) # 1 * b * h
        o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device = self.device) # 1 * b * h
        X = self.word_embedding(torch.tensor([self.start_idx] * self.batch_size, device = self.device)).unsqueeze(0) # 1 * b * e

        rewards = 0
        for t in range(1, self.max_length):
            output, (h_prev, o_prev) = self.LSTM(X, (h_prev, o_prev))
            logits = self.vocab_projection(output).squeeze(0) # b * v
            P = F.log_softmax(logits, dim = -1) # b * v
            word_t = fake_samples[ : , t] # b
            P_t = torch.gather(P, 1, word_t.unsqueeze(1)).squeeze(1) # b
            X = self.word_embedding(word_t).unsqueeze(0) # 1 * b * e

            self.eval()
            with torch.no_grad():
                monte_carlo_X = word_t.repeat_interleave(self.monte_carlo_num) # (b * M)
                monte_carlo_X = self.word_embedding(monte_carlo_X).unsqueeze(0) # 1 * (b * M) * e
                monte_carlo_h_prev = h_prev.clone().detach().repeat_interleave(self.monte_carlo_num, dim = 1) # 1 * (b * M) * h
                monte_carlo_o_prev = o_prev.clone().detach().repeat_interleave(self.monte_carlo_num, dim = 1) # 1 * (b * M) * h
                monte_carlo_output = torch.zeros(self.max_length, self.batch_size * self.monte_carlo_num, dtype = torch.long, device = self.device) # len * (b * M)

                for i in range(self.max_length - t - 1):
                    output, (monte_carlo_h_prev, monte_carlo_o_prev) = self.LSTM(monte_carlo_X, (monte_carlo_h_prev, monte_carlo_o_prev))
                    P = F.softmax(self.vocab_projection(output), dim = -1).squeeze(0) # (b * M) * v
                    for j in range(P.shape[0]):
                        monte_carlo_output[i + t + 1][j] = torch.multinomial(P[j], 1)[0]
                    monte_carlo_X = self.word_embedding(monte_carlo_output[i + t + 1]).unsqueeze(0) # 1 * (b * M) * e

                monte_carlo_output = monte_carlo_output.permute(1, 0) # (b * M) * len
                monte_carlo_output[ : , : t + 1] = fake_samples[ : , : t + 1].repeat_interleave(self.monte_carlo_num, dim = 0)
    
                discriminator_out = discriminator_func(monte_carlo_output) # (b * M)
                reward = discriminator_out.reshape(self.batch_size, self.monte_carlo_num).mean(dim = 1) # b
            
            self.train()
            mask = (word_t != self.pad_idx) & (word_t != self.end_idx)
            reward = reward * P_t * mask
            reward = reward[reward.nonzero(as_tuple = True)]
            if (reward.shape[0]):
                rewards += reward.mean()
    
        return -rewards