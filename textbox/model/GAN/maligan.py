# @Time   : 2020/11/13
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import UnconditionalGenerator
from textbox.module.Generator.maligan_generator import MaliGANGenerator
from textbox.module.Discriminator.maligan_discriminator import MaliGANDiscriminator
from textbox.model.init import xavier_normal_initialization


class MaliGAN(UnconditionalGenerator):

    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(MaliGAN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.gen_rnn_type = config['gen_rnn_type']
        self.dis_rnn_type = config['dis_rnn_type']
        self.dropout_ratio = config['dropout_ratio']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        
        self.generator = MaliGANGenerator(self.embedding_size, self.hidden_size, self.num_layers, self.gen_rnn_type, self.dropout_ratio)
        self.discriminator = MaliGANDiscriminator(self.embedding_size, self.hidden_size, self.num_layers, self.dis_rnn_type, self.dropout_ratio)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, eval_data):
        generate_corpus = []
        number_to_gen = 10
        idx2token = eval_data.idx2token
        for _ in range(number_to_gen):
            hidden_states = torch.zeros(self.num_dec_layers, 1, self.hidden_size).to(self.device)
            generate_tokens = []
            input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            for gen_idx in range(100):
                decoder_input = self.token_embedder(input_seq)
                outputs, hidden_states = self.generator(decoder_input, hidden_states)
                token_logits = self.vocab_linear(outputs)
                topv, topi = torch.log(F.softmax(token_logits, dim=-1) + 1e-12).data.topk(k=4)
                topi = topi.squeeze()
                token_idx = topi[0].item()
                if token_idx == self.eos_token_idx or gen_idx >= 100:
                    break
                else:
                    generate_tokens.append(idx2token[token_idx])
                    input_seq = torch.LongTensor([[token_idx]]).to(self.device)
            generate_corpus.append(generate_tokens)
        return generate_corpus

    def calculate_gen_pretrain_loss(self, corpus, epoch_idx=0):  # real data
        input_text = corpus['target_text'][:, :-1]  # batch_size * seq_len
        target_text = corpus['target_text'][:, 1:]  # batch_size * seq_len

        input_embeddings = self.dropout(self.token_embedder(input_text))  # batch_size * seq_len * embedding_size
        outputs, hidden_states = self.generator(input_embeddings)

        token_logits = self.vocab_linear(outputs)  # batch_size * seq_len * vocab_size
        token_logits = token_logits.view(-1, token_logits.size(-1))  # (batch_size * seq_len) * vocab_size
        target_text = target_text.contiguous().view(-1)  # (batch_size * seq_len) * 1

        loss = self.loss(token_logits, target_text)
        return loss

    def calculate_gen_adv_loss(self, corpus, epoch_idx=0):  # generated data
        input_text = corpus['target_text'][:, :-1]  # batch_size * seq_len
        target_text = corpus['target_text'][:, 1:]  # batch_size * seq_len

        rewards = self.get_mali_reward(target_text)  # batch_size * seq_len

        input_embeddings = self.dropout(self.token_embedder(input_text))  # batch_size * seq_len * embedding_size
        outputs, hidden_states = self.generator(input_embeddings)
        token_logits = self.vocab_linear(outputs)  # batch_size * seq_len * vocab_size

        target_onehot = F.one_hot(target_text, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(token_logits * target_onehot, dim=-1)  # batch_size * seq_len
        loss = -torch.sum(pred * rewards).item()

        return loss

    def calculate_dis_loss(self, corpus, epoch_idx=0):  # real data and generated data
        input_text = corpus['target_text'][:, :-1]  # batch_size * seq_len
        #target_text = corpus['target_text'][:, 1:]  # label: (batch_size)  To do: add target_label

        input_embeddings = self.dropout(self.token_embedder(input_text))  # batch_size * seq_len * embedding_size
        pred = self.discriminator(input_embeddings)  # batch_size * 2

        loss = self.loss(pred, target_text)

        return loss

    def get_mali_reward(self, samples):
        rewards = []
        rollout_num = 16  # note
        samples_embeddings = self.token_embedder(samples)  # batch_size * seq_len * embedding_size
        for _ in range(rollout_num):
            dis_out = F.softmax(self.discriminator(samples_embeddings), dim=-1)[:, 1]  # dis: batch_size * 2 #final: batch_size
            rewards.append(dis_out)

        rewards = torch.mean(torch.stack(rewards, dim=0), dim=0)  # batch_size
        rewards = torch.div(rewards, 1 - rewards)  # rD = D(x) / (1 - D(x))
        rewards = torch.div(rewards, torch.sum(rewards))
        rewards -= torch.mean(rewards) # To do: b?
        rewards = rewards.unsqueeze(1).expand(samples.size())  # batch_size * seq_len  # repeat

        return rewards