# @Time   : 2020/12/26
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization


class MaskGANGenerator(GenerativeAdversarialNet):
    r"""RNN-based Encoder-Decoder architecture for maskgan generator
    """

    def __init__(self, config, dataset):
        super(MaskGANGenerator, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.bidirectional = config['bidirectional']
        self.combine_method = config['combine_method']
        self.dropout_ratio = config['dropout_ratio']
        self.attention_type = config['attention_type']
        self.context_size = config['context_size']
        self.gamma = config['rl_discount_rate']
        self.advantage_clipping = config['advantage_clipping']
        self.eval_generate_num = config['eval_generate_num']
        self.attention_type = config['attention_type']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        self.mask_token_idx = dataset.token2idx[config["user_token_list"][0]]
        self.max_length = config['max_seq_length']
        self.embedder = nn.Embedding(self.vocab_size, self.embedding_size)

        # note!!! batch_first is true
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type,
                                       self.dropout_ratio, self.bidirectional)

        if self.attention_type is not None:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.hidden_size, self.context_size,
                                                 self.num_dec_layers, self.rnn_type, self.dropout_ratio,
                                                 self.attention_type)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.hidden_size, self.num_dec_layers,
                                           self.rnn_type, self.dropout_ratio)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mask_input(self, inputs, targets_present):
        r"""Transforms the inputs to have missing tokens when it's masked out.  The
        mask is for the targets, so therefore, to determine if an input at time t is
        masked, we have to check if the target at time t - 1 is masked out.

        e.g.

        - inputs = [a, b, c, d]

        - targets = [b, c, d, e]

        - targets_present = [1, 0, 1, 0]

        then,
        
        - masked_input = [a, b, <missing>, d]
        
        Args:
            inputs: Tensor of shape [batch_size, sequence_length]
            targets_present: Bool tensor of shape [batch_size, sequence_length] with
                             1 representing the presence of the word.
        
        Returns:
            masked_input: Tensor of shape [batch_size, sequence_length]
                          which takes on value of inputs when the input is present and takes on
                          value=mask_token_idx to indicate a missing token.
        """
        inputs_missing = torch.zeros_like(inputs)
        inputs_missing[:, :] = self.mask_token_idx

        zeroth_input_present = torch.ones_like(targets_present)[:, 1].unsqueeze(dim=-1)  # bs*1

        inputs_present = torch.cat([zeroth_input_present, targets_present], dim=1)[:, :-1]  # bs*seq_len

        masked_input = torch.where(inputs_present, inputs, inputs_missing)

        return masked_input

    def forward(self, inputs, input_length, targets, targets_present, pretrain=False, validate=False):
        r"""Input real padded input and target sentence which not start from sos and end with eos(According to origin
        code). And input length used for LSTM

        Args:
            inputs: bs*seq_len
            input_length:  list[bs]
            targets_present: target present matrix bs*seq_len 1: not mask 0: mask
            pretrain: control whether LM pretrain

        Returns:
            output: samples
            log_probs: log prob
            logits: logits
        """
        real_input = inputs
        masked_input = self.mask_input(inputs, targets_present)  # bs * seq_len

        masked_rnn_input = self.embedder(masked_input)  # bs * seq_len * emb

        if self.rnn_type == "lstm":
            # outputs: bs * seq_len * emb, state [h0/c0: 2*bs*hid]
            encoder_outputs, encoder_final_state = self.encoder(masked_rnn_input, input_length)
            h, c = encoder_final_state
            h = h.contiguous()
            c = c.contiguous()
            encoder_final_state = (h, c)
        else:
            encoder_outputs, encoder_final_state = self.encoder(masked_rnn_input, input_length)

        if pretrain:
            # if pretrain whcih means pretrain lstm like LM
            return encoder_outputs

        outputs = []  # output token
        log_probs = []  # the log prob of output token
        logits = []  # after vocab projection
        sample_t = None
        seq_len = inputs.size()[1]
        hidden_state = encoder_final_state
        for t in range(seq_len):
            if t == 0:  # Always provide the real input at t = 0.
                input_t = self.embedder(inputs[:, t].unsqueeze(dim=-1))  # bs*1*emb_dim
            elif validate:
                input_t = self.embedder(inputs[:, t].unsqueeze(dim=-1))
            else:
                real_input_t = self.embedder(inputs[:, t].unsqueeze(dim=-1))  # bs*1*emb_dim
                mask_input_t = self.embedder(sample_t)  # bs*1*emb_dim
                input_t = torch.where(targets_present[:, t - 1].unsqueeze(dim=1).unsqueeze(dim=2), real_input_t,
                                      mask_input_t)

            if self.attention_type is not None:
                encoder_mask = torch.ones_like(inputs)
                rnn_output, hidden_state, _ = self.decoder(input_t, hidden_state, encoder_outputs, encoder_mask)
            else:
                rnn_output, hidden_state = self.decoder(input_t, hidden_state)
            logit = self.vocab_linear(rnn_output)  # bs*1*vocab_size
            logits.append(logit)
            prob = torch.softmax(logit, dim=-1)  # bs*1*vocab_size
            categorical = Categorical(probs=prob)
            sample_t = categorical.sample()  # bs*1
            log_prob = categorical.log_prob(sample_t).squeeze(dim=-1)  # bs
            real_t = targets[:, t]  # bs
            output = torch.where(targets_present[:, t], real_t, sample_t.squeeze(dim=-1))  # bs
            outputs.append(output)
            log_probs.append(log_prob)

        outputs = torch.stack(outputs, dim=1)  # bs*seq_len
        log_probs = torch.stack(log_probs, dim=1)  # bs*seq_len
        logits = torch.stack(logits, dim=1).squeeze(dim=2)  # bs*seq_len*vocab_size
        return outputs, log_probs, logits

    def mask_cross_entropy_loss(self, targets, logits, targets_present):
        r"""Calculate the filling token cross entropy loss
        """
        targets = targets.long()
        cl = nn.CrossEntropyLoss(reduction="none")
        loss = cl(logits.permute([0, 2, 1]), targets)
        zeros_loss = torch.zeros_like(loss)
        missing_cl_loss = torch.where(targets_present, zeros_loss, loss)
        missing = 1 - targets_present.float()
        missing_cl_loss = torch.sum(missing_cl_loss) / torch.sum(missing)
        return missing_cl_loss

    def calculate_train_loss(self, inputs, lengths, targets, targets_present, validate=False):
        r"""Calculate train loss for generator
        """
        outputs, log_probs, logits = self.forward(inputs, lengths, targets, targets_present, validate=validate)
        loss = self.mask_cross_entropy_loss(targets, logits, targets_present)
        return loss

    def adversarial_loss(self, inputs, lengths, targets, targets_present, discriminator):
        r"""Calculate adversarial loss
        """
        outputs, log_probs, logits = self.forward(inputs, lengths, targets, targets_present)
        fake_predictions, _ = discriminator(inputs, lengths, outputs, targets_present, self.embedder)
        fake_predictions = fake_predictions.detach()
        est_state_values = discriminator.critic(inputs, outputs, self.embedder)
        rl_loss, critic_loss = self.calculate_reinforce_objective(log_probs, fake_predictions, targets_present,
                                                                  est_state_values)
        return (rl_loss, critic_loss)

    def create_critic_loss(self, cumulative_rewards, estimated_values, target_present):
        r"""Compute Critic loss in estimating the value function.  This should be an
        estimate only for the missing elements.
        """
        missing = 1. - target_present.float()
        cumulative_rewards = torch.where(target_present, estimated_values, cumulative_rewards)
        loss_f = nn.MSELoss(reduction='none')
        loss = loss_f(estimated_values, cumulative_rewards)
        mean_loss = torch.sum(loss) / torch.sum(missing)

        return mean_loss

    def calculate_reinforce_objective(self, log_probs, dis_predictions, mask_present, estimated_values=None):
        r"""Calculate the REINFORCE objectives.  The REINFORCE objective should only be on the tokens that were missing.
        Specifically, the final Generator reward should be based on the Discriminator predictions on missing tokens.
        The log probabilities should be only for missing tokens and the baseline should be calculated only on the missing
        tokens.
        For this model, we optimize the reward is the log of the *conditional* probability the Discriminator assigns to
        the distribution.  Specifically, for a Discriminator D which outputs probability of real, given the past context,
        r_t = log D(x_t|x_0,x_1,...x_{t-1})
        And the policy for Generator G is the log-probability of taking action x2 given the past context.
        
        Args:
            log_probs: Tensor of log probabilities of the tokens selected by the Generator.
                       Shape [batch_size, sequence_length].
            dis_predictions: Tensor of the predictions from the Discriminator.
                             Shape [batch_size, sequence_length].
            present: Tensor indicating which tokens are present.
                     Shape [batch_size, sequence_length].
            estimated_values: Tensor of estimated state values of tokens.
                              Shape [batch_size, sequence_length]
        
        Returns:
            final_gen_objective:  Final REINFORCE objective for the sequence.
            rewards:  Tensor of rewards for sequence of shape [batch_size, sequence_length]
            advantages: Tensor of advantages for sequence of shape [batch_size, sequence_length]
            baselines:  Tensor of baselines for sequence of shape [batch_size, sequence_length]
            maintain_averages_op:  ExponentialMovingAverage apply average op to maintain the baseline.
        """
        # Generator rewards are log-probabilities.
        eps = torch.tensor(1e-7)
        dis_predictions = torch.sigmoid(dis_predictions)
        rewards = torch.log(dis_predictions + eps)
        device = dis_predictions.device

        # Apply only for missing elements.
        zeros = torch.zeros_like(mask_present, dtype=torch.float32)
        log_probs = torch.where(mask_present, zeros, log_probs)
        rewards = torch.where(mask_present, zeros, rewards)
        rewards = rewards.detach()

        # Unstack Tensors into lists.
        missing = 1. - mask_present.float()

        # Cumulative Discounted Returns.  The true value function V*(s).
        cumulative_rewards = []
        batch_size, seq_len = dis_predictions.size()
        for t in range(seq_len):
            cum_value = torch.zeros((batch_size, 1))
            cum_value = cum_value.cuda(device)
            for s in range(t, seq_len):
                cum_value_tmp = missing[:, s] * np.power(self.gamma, (s - t)) * rewards[:, s]
                cum_value_tmp = cum_value_tmp.unsqueeze(dim=1)
                cum_value += cum_value_tmp
            cumulative_rewards.append(cum_value)  # bs*1
        cumulative_rewards = torch.stack(cumulative_rewards, dim=1).squeeze()

        # REINFORCE with different baselines.
        # We create a separate critic functionality for the Discriminator.  This
        # will need to operate unidirectionally and it may take in the past context.
        # Critic loss calculated from the estimated value function \hat{V}(s)
        # versus the true value function V*(s).
        cumulative_rewards = cumulative_rewards.detach()
        critic_loss = self.create_critic_loss(cumulative_rewards, estimated_values, mask_present)

        # Baselines are coming from the critic's estimated state values.
        baselines = estimated_values
        baselines = baselines.detach()

        ## Calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
        final_gen_objective = torch.zeros([batch_size, 1]).cuda(self.device)
        for t in range(seq_len):
            log_probability = log_probs[:, t].unsqueeze(dim=1)  # bs*1
            cum_advantage = torch.zeros((batch_size, 1))  # bs*1
            cum_advantage = cum_advantage.cuda(self.device)
            for s in range(t, seq_len):
                cum_advantage_tmp = missing[:, s] * np.power(self.gamma, (s - t)) * rewards[:, s]
                cum_advantage_tmp = cum_advantage_tmp.unsqueeze(dim=1)
                cum_advantage = cum_advantage + cum_advantage_tmp
            cum_advantage = cum_advantage - baselines[:, t].unsqueeze(dim=1)
            # Clip advantages.
            cum_advantage = torch.clamp(cum_advantage, -self.advantage_clipping, self.advantage_clipping)
            cum_advantage_ = cum_advantage.detach()
            final_gen_objective = final_gen_objective + torch.mul(log_probability,
                                                                  missing[:, t].unsqueeze(dim=1) * cum_advantage_)
        final_gen_objective = -torch.sum(final_gen_objective) / (torch.sum(missing))  # max the reward

        return final_gen_objective, critic_loss

    def calculate_loss(self, logits, target_inputs):
        r"""Calculate nll test loss
        """
        targets = target_inputs.long()  # bs*seq_len
        bs, seq_len = targets.size()
        targets = targets.contiguous()
        targets = targets.reshape((-1))  # (bs*seq_len)
        logits = logits.contiguous()
        logits = logits.reshape((-1, self.vocab_size))
        nll_losses = F.cross_entropy(logits, targets, reduction="none")
        nll_losses = nll_losses.reshape((-1, seq_len))
        nll_losses = torch.sum(nll_losses, dim=1)
        nll_losses = torch.mean(nll_losses)
        return nll_losses

    def generate(self, corpus):
        r"""Sample sentence
        """
        number_to_gen = self.eval_generate_num
        real_data = self._get_real_data(corpus)
        corpus_num, _ = real_data.size()
        corpus_batches = corpus_num // self.batch_size
        num_batch = number_to_gen // self.batch_size + 1 if number_to_gen != self.batch_size else 1
        samples = torch.zeros(num_batch * self.batch_size, self.max_length).long()  # larger than num_samples
        idx2token = corpus.idx2token

        for b in range(num_batch):
            while b >= corpus_batches:
                b = b - corpus_batches
            inputs = real_data[b * self.batch_size: (b + 1) * self.batch_size, :-1]
            targets = real_data[b * self.batch_size: (b + 1) * self.batch_size, 1:]
            inputs_length = torch.Tensor([self.max_length - 1] * self.batch_size).float()
            targets_present = torch.zeros((self.batch_size, self.max_length - 1)).byte()
            device = inputs.device
            inputs_length = inputs_length.cuda(device)
            targets_present = targets_present.cuda(device)

            sample, _, _ = self.forward(inputs, inputs_length, targets, targets_present)

            assert sample.shape == (self.batch_size, self.max_length - 1)
            sample = torch.cat([inputs[:, 0].unsqueeze(dim=-1), sample], dim=-1)
            samples[b * self.batch_size: (b + 1) * self.batch_size, :] = sample

        samples = samples[:number_to_gen, :-1]
        samples = samples.tolist()
        texts = []
        for sen in samples:
            text = []
            for w in sen:
                if w != self.eos_token_idx:
                    text.append(idx2token[w])
                else:
                    break
            texts.append(text)

        return texts

    def _add_eos(self, data, length):
        batch_size, pad_seq_len = data.size()
        padded_data = torch.full((batch_size, self.max_length), self.eos_token_idx, dtype=torch.long)
        device = data.device
        padded_data = padded_data.cuda(device)
        for i in range(batch_size):
            l = int(length[i].cpu().data)
            if l == self.max_length + 2:
                padded_data[i, :] = data[i, 1:l - 1]
            else:
                padded_data[i, 0:l - 1] = data[i, 1:l]
        return padded_data

    def _get_real_data(self, train_data):
        real_datas = []
        for corpus in train_data:
            real_data = corpus['target_idx']  # bs*batch_max_seq_len
            length = corpus['target_length']
            real_data = self._add_eos(real_data, length)
            real_datas.append(real_data)

        real_datas = torch.cat(real_datas, dim=0)
        return real_datas
