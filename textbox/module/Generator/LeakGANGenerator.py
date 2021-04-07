# @Time   : 2020/11/19
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn

r"""
LeakGAN Generator
#####################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from textbox.model.abstract_generator import UnconditionalGenerator
from torch.distributions import Categorical


class LeakGANGenerator(UnconditionalGenerator):
    r"""LeakGAN generator consist of worker(LSTM) and manager(LSTM)
    """

    def __init__(self, config, dataset):
        super(LeakGANGenerator, self).__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.embedding_size = config['generator_embedding_size']
        self.max_length = config['max_seq_length'] + 1  # max_length is the length of origin_max_len + sos
        self.monte_carlo_num = config['Monte_Carlo_num']
        self.filter_nums = config['filter_nums']
        self.goal_out_size = sum(self.filter_nums)
        self.goal_size = config['goal_size']
        self.step_size = config['step_size']
        self.temperature = config['temperature']
        self.dis_sample_num = config['d_sample_num']
        self.start_idx = dataset.sos_token_idx
        self.end_idx = dataset.eos_token_idx
        self.pad_idx = dataset.padding_token_idx
        self.use_gpu = config['use_gpu']
        self.gpu_id = config['gpu_id']

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.vocab_projection = nn.Linear(self.hidden_size, self.vocab_size)

        self.worker = nn.LSTM(self.embedding_size, self.hidden_size)
        self.manager = nn.LSTM(self.goal_out_size, self.hidden_size)

        self.work2goal = nn.Linear(self.hidden_size, self.vocab_size * self.goal_size)
        self.mana2goal = nn.Linear(self.hidden_size, self.goal_out_size)
        self.goal2goal = nn.Linear(self.goal_out_size, self.goal_size, bias=False)

        self.goal_init = nn.Parameter(torch.rand((self.batch_size, self.goal_out_size)))

    def pretrain_loss(self, corpus, dis):
        r"""Return the generator pretrain loss for predicting target sequence.

        Args:
            corpus: target_text(bs*seq_len)
            dis: discriminator model

        Returns:
            manager_loss: manager loss
            work_cn_loss: worker loss
        """
        targets = corpus[:, 1:]  # not use sos token
        batch_size, seq_len = targets.size()  # bs*max_seq_len
        leak_out_array, feature_array, goal_array = self.leakgan_forward(targets, dis, train=False, pretrain=True)

        # Manager loss
        mana_cos_loss = self.manager_cos_loss(
            batch_size, feature_array, goal_array
        )  # batch_size * (seq_len / step_size)
        manager_loss = -torch.sum(mana_cos_loss) / (self.batch_size * self.max_length / self.step_size)
        # Worker loss
        work_cn_loss = self.worker_cross_entropy_loss(targets, leak_out_array)

        return manager_loss, work_cn_loss

    def calculate_loss(self, targets, dis):
        r"""Returns the nll test for predicting target sequence.

        Args:
            targets: target_idx(bs*seq_len) ,
            dis: discriminator model

        Returns:
            worker_loss: the generator test nll
        """
        batch_size, seq_len = targets.size()
        leak_out_array, feature_array, goal_array = self.leakgan_forward(targets, dis, pretrain=True)

        # Worker loss
        work_nll_loss = self.worker_nll_loss(targets, leak_out_array)  # batch_size * seq_len
        work_nll_loss = work_nll_loss.contiguous().reshape((batch_size, seq_len))
        work_nll_loss = torch.sum(work_nll_loss, dim=1)  # bs
        worker_loss = torch.mean(work_nll_loss)

        return worker_loss

    def forward(self, idx, inp, work_hidden, mana_hidden, feature, real_goal, train=False, pretrain=False):
        r"""Embed input and sample on token at a time (seq_len = 1)

        Args:
            idx: index of current token in sentence
            inp: current input token for a batch [batch_size]
            work_hidden: 1 * batch_size * hidden_dim
            mana_hidden: 1 * batch_size * hidden_dim
            feature: 1 * batch_size * total_num_filters, feature of current sentence
            real_goal: batch_size * goal_out_size, real_goal in LeakGAN source code
            train: whether train or inference
            pretrain: whether pretrain or not pretrain

        Returns:
            out: current output prob over vocab with log_softmax or softmax bs*vocab_size
            cur_goal: bs * 1 * goal_out_size
            work_hidden: 1 * batch_size * hidden_dim
            mana_hidden: 1 * batch_size * hidden_dim
        """
        emb = self.word_embedding(inp).unsqueeze(0)  # 1 * batch_size * embed_dim

        # Manager
        mana_out, mana_hidden = self.manager(feature, mana_hidden)  # mana_out: 1 * batch_size * hidden_dim
        mana_out = self.mana2goal(mana_out.permute([1, 0, 2]))  # batch_size * 1 * goal_out_size
        cur_goal = F.normalize(mana_out, p=2, dim=-1).squeeze(dim=1)
        _real_goal = self.goal2goal(real_goal)  # batch_size * goal_size
        _real_goal = F.normalize(_real_goal, p=2, dim=-1).unsqueeze(-1)  # batch_size * goal_size * 1

        # Worker
        work_out, work_hidden = self.worker(emb, work_hidden)  # work_out: 1 * batch_size * hidden_dim
        work_out = self.work2goal(work_out.squeeze(dim=0))  # bs * (vocab*goal)
        work_out = work_out.contiguous().view(
            -1, self.vocab_size, self.goal_size
        )  # batch_size * vocab_size * goal_size

        # Sample token
        out = torch.matmul(work_out, _real_goal).squeeze(-1)  # batch_size * vocab_size

        # Temperature control
        if idx > 1:
            if train:  # if train we should use a min temperature to modify the out distribution
                temperature = 0.9
            else:
                temperature = self.temperature
        else:
            temperature = self.temperature

        if not pretrain:
            out = temperature * out  # bs * vocab

        return out, cur_goal, work_hidden, mana_hidden

    def leakgan_forward(self, targets, dis, train=False, pretrain=False):
        r"""Get all feature and goals according to given sentences

        Args:
            targets: batch_size * max_seq_len, pad eos token if the original sentence length less than max_seq_len
            dis: discriminator model
            train: if use temperature parameter
            pretrain: whether pretrain or not pretrain

        Returns:
            feature_array: batch_size * (seq_len + 1) * total_num_filter
            goal_array: batch_size * (seq_len + 1) * goal_out_size
            leak_out_array: batch_size * seq_len * vocab_size with log_softmax
        """
        batch_size, seq_len = targets.size()  # seq_len = max_seq_len

        feature_array = torch.zeros((batch_size, self.max_length + 1, self.goal_out_size))
        goal_array = torch.zeros((batch_size, self.max_length + 1, self.goal_out_size))
        leak_out_array = torch.zeros((batch_size, self.max_length + 1, self.vocab_size))
        if self.use_gpu:
            feature_array = feature_array.cuda(self.gpu_id)
            goal_array = goal_array.cuda(self.gpu_id)
            leak_out_array = leak_out_array.cuda(self.gpu_id)

        work_hidden = self.init_hidden(batch_size)
        mana_hidden = self.init_hidden(batch_size)
        # Special operations for step 0
        leak_inp_t = torch.LongTensor([self.start_idx] * batch_size)  # the input token for worker at step t
        cur_dis_inp = torch.LongTensor([self.pad_idx] * batch_size * seq_len)  # current sentence for dis ar step t
        cur_dis_inp = cur_dis_inp.view((batch_size, seq_len))  # bs*seq_len
        if self.use_gpu:
            leak_inp_t = leak_inp_t.cuda(self.gpu_id)
            cur_dis_inp = cur_dis_inp.cuda(self.gpu_id)

        real_goal = self.goal_init[:batch_size, :]  # init real goal
        goal_array[:, 0, :] = real_goal
        last_goal = torch.zeros_like(real_goal)
        feature = dis.get_feature(cur_dis_inp).unsqueeze(0)  # !!!note: 1 * batch_size * total_num_filters
        feature_array[:, 0, :] = feature.squeeze(0)  # batch_size * total_num_filters
        # Update the hidden state of manager using the current all padding token
        _, mana_hidden = self.manager(feature, mana_hidden)  # mana_out: 1 * batch_size * hidden_dim

        for i in range(1, self.max_length + 1):
            # get current dis inp which giving the real top i token and padding token
            given_dis_inp = targets[:, :i]  # bs*i
            cur_dis_inp = torch.cat([given_dis_inp, cur_dis_inp], dim=1)
            cur_dis_inp = cur_dis_inp[:, :seq_len].long()
            # get feature
            feature = dis.get_feature(cur_dis_inp).unsqueeze(0)  # !!!note: 1 * batch_size * total_num_filters
            feature_array[:, i, :] = feature.squeeze(0)  # batch_size * total_num_filters
            # using input_t and feature_t to get token_t+1
            # out is the log softmax over vocab distribution
            out, cur_goal, work_hidden, mana_hidden = self.forward(
                i, leak_inp_t, work_hidden, mana_hidden, feature, real_goal, train=train, pretrain=pretrain
            )
            leak_out_array[:, i - 1, :] = out
            # save the current goal_t
            goal_array[:, i, :] = cur_goal

            # update real goal every step_size steps
            if i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)  # g1 -> g4

            # use the real input token during train
            leak_inp_t = targets[:, i - 1]
            if self.use_gpu:
                leak_inp_t = leak_inp_t.cuda(self.gpu_id)
        # cur to seq_len
        leak_out_array = leak_out_array[:, :seq_len, :]

        return leak_out_array, feature_array, goal_array

    def sample_batch(self):
        r"""Sample a batch of data
        """
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            o_prev = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)  # 1 * b * h
            prev_state = (h_prev, o_prev)
            X = self.word_embedding(
                torch.tensor([self.start_idx] * self.batch_size, dtype=torch.long, device=self.device)
            ).unsqueeze(0)  # 1 * b * e
            sentences = torch.zeros((self.max_length, self.batch_size), dtype=torch.long, device=self.device)
            sentences[0] = self.start_idx

            for i in range(1, self.max_length):
                output, prev_state = self.LSTM(X, prev_state)
                P = F.softmax(self.vocab_projection(output), dim=-1).squeeze(0)  # b * v
                for j in range(self.batch_size):
                    sentences[i][j] = torch.multinomial(P[j], 1)[0]
                X = self.word_embedding(sentences[i]).unsqueeze(0)  # 1 * b * e

            sentences = sentences.permute(1, 0)  # b * l

            for i in range(self.batch_size):
                end_pos = (sentences[i] == self.end_idx).nonzero(as_tuple=False)
                if (end_pos.shape[0]):
                    sentences[i][end_pos[0][0] + 1:] = self.pad_idx

        self.train()

        return sentences

    def sample(self, sample_num, dis, start_letter, train=False):
        r"""Sample sentences
        """
        num_batch = sample_num // self.batch_size + 1 if sample_num != self.batch_size else 1
        samples = torch.zeros(num_batch * self.batch_size, self.max_length).long()  # larger than num_samples
        fake_sentences = torch.zeros((self.batch_size, self.max_length))
        fake_sentences[:, :] = self.pad_idx

        for b in range(num_batch):
            leak_sample = self.leakgan_generate(fake_sentences, dis, train=train)

            assert leak_sample.shape == (self.batch_size, self.max_length)
            samples[b * self.batch_size:(b + 1) * self.batch_size, :] = leak_sample

        samples = samples[:sample_num, :]
        if self.use_gpu:
            samples = samples.cuda(self.gpu_id)

        return samples

    def leakgan_generate(self, targets, dis, train=False):
        batch_size, seq_len = targets.size()
        samples = []
        log_probs = []

        work_hidden = self.init_hidden(batch_size)
        mana_hidden = self.init_hidden(batch_size)

        real_goal = self.goal_init[:batch_size, :]  # init real goal
        last_goal = torch.zeros_like(real_goal)

        for i in range(0, self.max_length):
            if i == 0:
                leak_inp_t = torch.LongTensor([self.start_idx] * batch_size)  # the input token for worker at step t
                cur_dis_inp = torch.LongTensor([self.pad_idx] * batch_size * seq_len
                                               )  # current sentence for dis ar step t
                cur_dis_inp = cur_dis_inp.view((batch_size, seq_len))  # bs*seq_len
            else:
                leak_inp_t = gen_x
                cur_dis_inp = torch.cat([gen_x.unsqueeze(dim=1), cur_dis_inp], dim=-1)
                cur_dis_inp = cur_dis_inp[:, :self.max_length].long()
            if self.use_gpu:
                leak_inp_t = leak_inp_t.cuda(self.gpu_id)
                cur_dis_inp = cur_dis_inp.cuda(self.gpu_id)

            # get feature
            feature = dis.get_feature(cur_dis_inp).unsqueeze(0)  # !!!note: 1 * batch_size * total_num_filters

            # using input_t and feature_t to get token_t+1
            # out is the softmax over vocab distribution
            out, cur_goal, work_hidden, mana_hidden = self.forward(
                i, leak_inp_t, work_hidden, mana_hidden, feature, real_goal, train=train, pretrain=False
            )
            out_dis = Categorical(F.softmax(out, dim=-1))  # bs * vocab
            gen_x = out_dis.sample()  # bs
            gen_x_prob = out_dis.log_prob(gen_x)
            samples.append(gen_x)
            log_probs.append(gen_x_prob)

            last_goal = last_goal + cur_goal

            # update real goal every step_size steps
            if (i + 1) % self.step_size == 0:
                real_goal = last_goal
                last_goal = torch.zeros_like(real_goal)

        samples = torch.stack(samples, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        if self.use_gpu:
            samples = samples.cuda(self.gpu_id)
            log_probs = log_probs.cuda(self.gpu_id)
        return samples

    def generate(self, batch_data, eval_data, dis):
        r"""Generate sentences
        """
        fake_sentences = torch.zeros((self.batch_size, self.max_length))
        idx2token = eval_data.idx2token
        batch_size = len(batch_data['target_text'])

        samples = self.leakgan_generate(fake_sentences, dis)
        samples = samples[:batch_size]
        samples = samples.tolist()
        texts = []
        for sen in samples:
            text = []
            for w in sen:
                if w != self.end_idx:
                    text.append(idx2token[w])
                else:
                    break
            texts.append(text)

        return texts

    def adversarial_loss(self, dis):
        r"""Generate data and calculate adversarial loss
        """
        with torch.no_grad():
            gen_samples = self.sample(
                self.batch_size, dis, self.start_idx, train=True
            )  # !!! train=True, the only place

        rewards = self.get_reward_leakgan(gen_samples, self.monte_carlo_num, dis).cpu()  # reward with MC search
        mana_loss, work_loss = self.get_adv_loss(gen_samples, rewards, dis)

        return (mana_loss, work_loss)

    def init_hidden(self, batch_size=1):
        r"""Init hidden state for lstm
        """
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)

        if self.use_gpu:
            h = h.cuda(self.gpu_id)
            c = c.cuda(self.gpu_id)
        return h, c

    def manager_cos_loss(self, batch_size, feature_array, goal_array):
        """Get manager cosine distance loss

        Returns:
            cos_loss: batch_size * (seq_len / step_size)
        """
        sub_feature = torch.zeros(batch_size, self.max_length // self.step_size, self.goal_out_size)
        real_goal = torch.zeros(batch_size, self.max_length // self.step_size, self.goal_out_size)
        for i in range(self.max_length // self.step_size):
            idx = i * self.step_size  # 0, 4, 8, 16, 20
            sub_feature[:, i, :] = feature_array[:, idx + self.step_size, :] - feature_array[:, idx, :]

            if i == 0:
                real_goal[:, i, :] = goal_array[:, i, :]
            else:
                idx = (i - 1) * self.step_size + 1
                real_goal[:, i, :] = torch.sum(goal_array[:, idx:idx + 4, :], dim=1)

        # L2 noramlization
        sub_feature = F.normalize(sub_feature, p=2, dim=-1)
        real_goal = F.normalize(real_goal, p=2, dim=-1)

        cos_loss = F.cosine_similarity(sub_feature, real_goal, dim=-1)

        return cos_loss

    def worker_cross_entropy_loss(self, target, leak_out_array, reduction='mean'):
        r"""Get CrossEntropy loss for worker
        """
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        leak_out_array = leak_out_array.contiguous()
        leak_out_array = leak_out_array.reshape((-1, self.vocab_size))
        target = target.contiguous()
        target = target.reshape((-1))
        loss = loss_fn(leak_out_array, target)

        return loss

    def worker_nll_loss(self, target, leak_out_array):
        r"""Get nll loss for worker
        """
        loss_fn = nn.NLLLoss(reduction='none')
        leak_out_array = leak_out_array.contiguous()
        leak_out_array = leak_out_array.reshape((-1, self.vocab_size))
        leak_out_array = torch.log_softmax(leak_out_array, dim=-1)
        target = target.contiguous()
        target = target.reshape((-1))

        loss = loss_fn(leak_out_array, target)

        return loss

    def worker_cos_reward(self, feature_array, goal_array):
        """Get reward for worker (cosine distance)

        Returns:
            cos_loss: batch_size * seq_len
        """
        for i in range(self.max_length // self.step_size):
            real_feature = feature_array[:, i * self.step_size, :].unsqueeze(1).expand((-1, self.step_size, -1))
            feature_array[:, i * self.step_size:(i + 1) * self.step_size, :] = real_feature
            if i > 0:
                sum_goal = torch.sum(goal_array[:, (i - 1) * self.step_size:i * self.step_size, :], dim=1, keepdim=True)
            else:
                sum_goal = goal_array[:, 0, :].unsqueeze(1)
            goal_array[:, i * self.step_size:(i + 1) * self.step_size, :] = sum_goal.expand((-1, self.step_size, -1))

        offset_feature = feature_array[:, 1:, :]  # f_{t+1}, batch_size * seq_len * goal_out_size
        goal_array = goal_array[:, :self.max_length, :]  # batch_size * seq_len * goal_out_size
        sub_feature = offset_feature - goal_array

        # L2 normalization
        sub_feature = F.normalize(sub_feature, p=2, dim=-1)
        all_goal = F.normalize(goal_array, p=2, dim=-1)

        cos_loss = F.cosine_similarity(sub_feature, all_goal, dim=-1)  # batch_size * seq_len
        return cos_loss

    def split_params(self):
        r"""Split parameter into worker and manager
        """
        mana_params = list()
        work_params = list()

        mana_params += list(self.manager.parameters())
        mana_params += list(self.mana2goal.parameters())
        mana_params.append(self.goal_init)

        work_params += list(self.word_embedding.parameters())
        work_params += list(self.worker.parameters())
        work_params += list(self.work2goal.parameters())
        work_params += list(self.goal2goal.parameters())

        return mana_params, work_params

    def get_reward_leakgan(self, sentences, rollout_num, dis, current_k=0):
        r"""Get reward via Monte Carlo search for LeakGAN

        Args:
            sentences: size of batch_size * max_seq_len
            rollout_num: numbers of rollout
            dis: discriminator
            current_k: current training gen

        Returns:
            reward: batch_size * (max_seq_len / step_size)
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * (self.max_length // self.step_size), batch_size]).float()
            if self.use_gpu:
                rewards = rewards.cuda(self.gpu_id)
            idx = 0
            for i in range(rollout_num):
                for t in range(1, self.max_length // self.step_size):
                    given_num = t * self.step_size  # 4, 8, 12, ..
                    # given current words and search a complete sentence by mc
                    samples = self.rollout_mc_search_leakgan(sentences, dis, given_num)
                    out = dis(samples)  # bs*2
                    out = F.softmax(out, dim=-1)
                    # using the prob of true computed by dis as the reward for current action reward
                    reward = out[:, current_k + 1]  # bs
                    rewards[idx] = reward
                    idx += 1

                last_token_out = dis(sentences)
                last_token_out = F.softmax(last_token_out, dim=-1)
                last_token_reward = last_token_out[:, current_k + 1]
                rewards[idx] = last_token_reward
                idx += 1

        rewards = rewards.contiguous()
        rewards = rewards.view(batch_size, self.max_length // self.step_size, rollout_num)
        rewards = torch.sum(rewards, dim=-1)
        rewards_ = torch.mean(rewards, dim=-1)
        rewards = self.rescale(rewards, rollout_num)
        rewards = rewards / (1.0 * rollout_num)
        # rewards = torch.mean(rewards, dim=-1)
        return rewards

    def rescale(self, reward, rollout_num=1.0):
        r"""Rescale reward according to original paper
        """
        ret = torch.zeros_like(reward)
        reward = reward.cpu().numpy()
        x, y = reward.shape
        for i in range(x):
            l = reward[i]
            rescalar = {}
            for s in l:
                rescalar[s] = s
            idxx = 1
            min_s = 1.0
            max_s = 0.0
            for s in rescalar:
                rescalar[s] = self.redistribution(idxx, len(l), min_s)
                idxx += 1
            for j in range(y):
                ret[i, j] = rescalar[reward[i, j]]
        return ret

    def redistribution(self, idx, total, min_v):
        idx = (idx + 0.0) / (total + 0.0) * 16.0
        return (math.exp(idx - 8.0) / (1.0 + math.exp(idx - 8.0)))

    def rollout_mc_search_leakgan(self, targets, dis, given_num):
        r"""Roll out to get mc search results
        """
        batch_size, seq_len = targets.size()
        work_hidden = self.init_hidden(batch_size)
        mana_hidden = self.init_hidden(batch_size)
        real_goal = self.goal_init[:batch_size, :]
        out = 0

        leak_inp_t = torch.LongTensor([self.start_idx] * batch_size)  # the input token for worker at step t
        cur_dis_inp = torch.LongTensor([self.pad_idx] * batch_size * seq_len)  # current sentence for dis ar step t
        cur_dis_inp = cur_dis_inp.view((batch_size, seq_len))  # bs*seq_len
        leak_out_array = []
        if self.use_gpu:
            leak_inp_t = leak_inp_t.cuda(self.gpu_id)
            cur_dis_inp = cur_dis_inp.cuda(self.gpu_id)
            targets = targets.cuda(self.gpu_id)

        real_goal = self.goal_init[:batch_size, :]  # init real goal
        last_goal = torch.zeros_like(real_goal)
        feature = dis.get_feature(cur_dis_inp).unsqueeze(0)  # !!!note: 1 * batch_size * total_num_filters
        # Update the hidden state of manager using the current all padding token
        _, mana_hidden = self.manager(feature, mana_hidden)  # mana_out: 1 * batch_size * hidden_dim

        # get current state
        for i in range(1, given_num + 1):
            # get current dis inp which giving the real top i token and padding token
            given_dis_inp = targets[:, :i]  # bs*i
            cur_dis_inp = torch.cat([given_dis_inp, cur_dis_inp], dim=1)
            cur_dis_inp = cur_dis_inp[:, :seq_len].long()
            # get feature
            feature = dis.get_feature(cur_dis_inp).unsqueeze(0)  # !!!note: 1 * batch_size * total_num_filters
            # using input_t and feature_t to get token_t+1
            # out is the log softmax over vocab distribution
            out, cur_goal, work_hidden, mana_hidden = self.forward(
                i, leak_inp_t, work_hidden, mana_hidden, feature, real_goal, train=False, pretrain=False
            )

            leak_out_array.append(targets[:, i - 1])

            last_goal = last_goal + cur_goal
            leak_inp_t = targets[:, i - 1]
            if self.use_gpu:
                leak_inp_t = leak_inp_t.cuda(self.gpu_id)

            # update real goal every step_size steps
            if i % self.step_size == 0:
                real_goal = last_goal
                last_goal = torch.zeros_like(real_goal)

        # MC search
        for i in range(given_num + 1, self.max_length + 1):
            # get the generated token
            gen_x = torch.stack(leak_out_array, dim=-1)
            if self.use_gpu:
                gen_x = gen_x.cuda(self.gpu_id)

            cur_dis_inp = torch.cat([gen_x, targets], dim=-1)
            cur_dis_inp = cur_dis_inp[:, :seq_len].long()
            # get feature
            feature = dis.get_feature(cur_dis_inp).unsqueeze(0)  # !!!note: 1 * batch_size * total_num_filters
            # using input_t and feature_t to get token_t+1
            # out is the log softmax over vocab distribution
            out, cur_goal, work_hidden, mana_hidden = self.forward(
                i, leak_inp_t, work_hidden, mana_hidden, feature, real_goal, train=True, pretrain=False
            )

            # sample one token
            out_dis = Categorical(F.softmax(out))
            leak_inp_t = out_dis.sample()  # bs

            if self.use_gpu:
                leak_inp_t = leak_inp_t.cuda(self.gpu_id)
            leak_out_array.append(leak_inp_t)

            last_goal = last_goal + cur_goal
            # update real goal every step_size steps
            if i % self.step_size == 0:
                real_goal = last_goal
                last_goal = torch.zeros_like(real_goal)

        gen_x = torch.stack(leak_out_array, dim=-1)
        gen_x = gen_x[:, :seq_len]
        if self.use_gpu:
            gen_x = gen_x.cuda(self.gpu_id)

        return gen_x

    def get_adv_loss(self, target, rewards, dis):
        r"""Return a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Args: target, rewards, dis, start_letter
            target: batch_size * seq_len
            rewards: batch_size * seq_len (discriminator rewards for each token)
        """
        batch_size, seq_len = target.size()
        leak_out_array, feature_array, goal_array = self.leakgan_forward(target, dis, train=True)

        # Manager Loss
        mana_cos_loss = self.manager_cos_loss(
            batch_size, feature_array, goal_array
        )  # batch_size * (seq_len / step_size)
        mana_loss = -torch.mean(rewards * mana_cos_loss)

        # Worker Loss
        work_cn_loss = self.worker_cross_entropy_loss(target, leak_out_array, reduction='none')  # batch_size * seq_len
        work_cos_reward = self.worker_cos_reward(feature_array, goal_array)  # batch_size * seq_len
        work_cos_reward = work_cos_reward.contiguous().reshape((-1))
        work_loss = -torch.mean(work_cn_loss * work_cos_reward)

        return mana_loss, work_loss
