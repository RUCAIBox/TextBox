import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from textbox.utils import InputType
from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization


class MaskGANGenerator(GenerativeAdversarialNet):
    r"""RNN-based Encoder-Decoder architecture is a basic framework for conditional text generation.

    """

    # input_type = InputType.PAIRTEXT

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

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx
        self.mask_token_idx = 3  # dataset.mask_token_idx

        # define layers and loss
        # TODO: should fix this bug
        self.source_vocab_size = self.vocab_size
        self.target_vocab_size = self.vocab_size
        self.source_token_embedder = nn.Embedding(self.source_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.target_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)

        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type,
                                       self.dropout_ratio, self.bidirectional, self.combine_method)

        if self.attention_type is not None:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.hidden_size, self.context_size,
                                                 self.num_dec_layers, self.rnn_type, self.dropout_ratio,
                                                 self.attention_type)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.hidden_size, self.num_dec_layers,
                                           self.rnn_type, self.dropout_ratio)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.target_vocab_size)
        # self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mask_input(self, inputs, targets_present):
        r"""Transforms the inputs to have missing tokens when it's masked out.  The
        mask is for the targets, so therefore, to determine if an input at time t is
        masked, we have to check if the target at time t - 1 is masked out.

        e.g.
          inputs = [a, b, c, d]
          targets = [b, c, d, e]
          targets_present = [1, 0, 1, 0]
        then,
          masked_input = [a, b, <missing>, d]

        Args:
          inputs:  Tensor of shape [batch_size, sequence_length]
          targets_present:  Bool tensor of shape [batch_size, sequence_length] with
            1 representing the presence of the word.

        Returns:
          masked_input:  Tensor of shape [batch_size, sequence_length]
            which takes on value of inputs when the input is present and takes on
            value=mask_token_idx to indicate a missing token.
        """
        # 创建一个跟inputs同样大小全为mask的变量
        inputs_missing = torch.zeros_like(inputs)

        # 第0个输入通常不会被mask，创建大小为bs*1的全1矩阵
        zeroth_input_present = torch.ones_like(targets_present)[:, 1].unsqueeze(dim=-1) # bs*1

        # 将第0个输入和target_present表示的剩余输入一起拼接成bs*seq_len的mask矩阵
        inputs_present = torch.cat([zeroth_input_present, targets_present], dim=1)[:, :-1] # bs*seq_len

        # 使用where函数根据inputs_present确定是否maskinputs
        masked_input = torch.where(inputs_present, inputs, inputs_missing)

        return masked_input

    def forward(self, inputs, input_length, targets, targets_present, is_advtrain=True):
        r"""输入真实的补齐的句子以及句子长度以及target_mask矩阵，通过seq2seq输出预测生成的target输出

        Args:
            inputs: 真实句子 bs*seq_len
            input_length:  句子长度 list[bs]
            targets_present: 目标句子的mask矩阵 bs*seq_len
            is_advtrain: 控制训练还是验证
        Returns:
            output: 生成的样本
            log_probs: 采样概率
            logits: 得分
        """
        # 得到real_input和mask_input
        real_input = inputs
        masked_input = self.mask_input(inputs, targets_present)

        # 将input转换为embedding
        real_rnn_input = self.source_token_embedder(real_input)
        masked_rnn_input = self.source_token_embedder(masked_input)

        # 将masked_rnn_input传入encoder得到encoder_final_state和encoder_outputs
        encoder_outputs, encoder_final_state = self.encoder(masked_rnn_input, input_length)

        # 使用decoder依次传入masked_input解码生成输出，注意如果在验证或者预训练时，使用teacher forcing机制
        # 如果不是则根据targets_present选择采样还是真实作为当前时刻的输入
        outputs = []
        log_probs = []
        logits = []
        sample_t = None
        seq_len = inputs.size()[1]
        hidden_state = encoder_final_state
        for t in range(seq_len):
            if t == 0:
                input_t = self.target_token_embedder(inputs[:, t].unsqueeze(dim=-1))    # bs*1*emb_dim
            else:
                real_input_t = self.target_token_embedder(inputs[:, t].unsqueeze(dim=-1))   # bs*1*emb_dim
                mask_input_t = self.target_token_embedder(sample_t)
                if not is_advtrain:
                    input_t = real_input_t
                else:
                    input_t = torch.where(targets_present[:, t - 1].unsqueeze(dim=1).unsqueeze(dim=2), real_input_t, mask_input_t)

            # rnn_output, hidden_state = self.decoder(input_t, hidden_state, encoder_outputs)
            rnn_output, hidden_state = self.decoder(input_t, hidden_state)  # need to fix attention decoder
            logit = self.vocab_linear(rnn_output)   # bs*1*vocab_size
            categorical = Categorical(logits=logit)
            sample_t = categorical.sample()     # bs*1
            log_prob = categorical.log_prob(sample_t).squeeze(dim=-1)   # bs
            real_t = targets[:, t]  # bs
            output = torch.where(targets_present[:, t], real_t, sample_t.squeeze(dim=-1))   # bs

            outputs.append(output)
            log_probs.append(log_prob)
            logits.append(logit)
        # 收集decoder解码过程中的output, log_probs, logits
        outputs = torch.stack(outputs, dim=1)   # bs*seq_len
        log_probs = torch.stack(log_probs, dim=1)   # bs*seq_len
        logits = torch.stack(logits, dim=1).squeeze(dim=2)  # bs*seq_len*vocab_size
        return outputs, log_probs, logits

    def create_masked_cross_entropy_loss(self, targets, present, logits):
        r"""Calculate the cross entropy loss matrices for the masked tokens."""

        targets = targets.long()    # bs*seq_len
        logits = logits.permute(0, 2, 1)  # bs*vocab_size*seq_len
        cross_entropy_losses = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.padding_token_idx)

        # Zeros matrix.
        zeros_losses = torch.zeros_like(cross_entropy_losses)
        missing_ce_loss = torch.where(present, cross_entropy_losses, zeros_losses)
        missing_ce_loss = torch.sum(missing_ce_loss) / torch.sum(present)  # get scalar
        return missing_ce_loss

    def calculate_train_loss(self, inputs, lengths, targets, targets_present, is_advtrain):
        outputs, log_probs, logits = self.forward(inputs, lengths, targets, targets_present, is_advtrain=is_advtrain)
        losses = self.create_masked_cross_entropy_loss(targets, targets_present, logits)

        return losses

    def adversarial_loss(self, inputs, lengths, targets, targets_present, discriminator):
        outputs, log_probs, logits = self.forward(inputs, lengths, targets, targets_present, is_advtrain=True)
        fake_predictions = discriminator(inputs, lengths, outputs, targets_present)
        est_state_values = discriminator.critic(outputs)
        rl_loss, critic_loss = self.calculate_reinforce_objective(log_probs, fake_predictions, targets_present, est_state_values)
        return (rl_loss, critic_loss)

    def create_critic_loss(self, cumulative_rewards, estimated_values, target_present):
        r"""Compute Critic loss in estimating the value function.  This should be an
        estimate only for the missing elements.

        """
        # missing = torch.cast(target_present, torch.int32)
        # missing = 1 - missing
        # missing = torch.cast(missing, torch.bool)
        estimated_values = torch.where(target_present, cumulative_rewards, estimated_values)
        loss = nn.MSELoss()
        l = loss(estimated_values, cumulative_rewards)

        return l

    def calculate_reinforce_objective(self, log_probs, dis_predictions, present, estimated_values=None):
        r"""Calculate the REINFORCE objectives.  The REINFORCE objective should only be on the tokens that were missing.
        Specifically, the final Generator reward should be based on the Discriminator predictions on missing tokens.
        The log probabilities should be only for missing tokens and the baseline should be calculated only on the missing
        tokens.

        For this model, we optimize the reward is the log of the *conditional* probability the Discriminator assigns to
        the distribution.  Specifically, for a Discriminator D which outputs probability of real, given the past context,
        r_t = log D(x_t|x_0,x_1,...x_{t-1})
        And the policy for Generator G is the log-probability of taking action x2 given the past context.


        Args:
            log_probs: Tensor of log probailities of the tokens selected by the Generator.
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

        # Apply only for missing elements.
        zeros = torch.zeros_like(present, dtype=torch.float32)
        log_probs = torch.where(present, zeros, log_probs)
        rewards = torch.where(present, zeros, rewards)

        # Unstack Tensors into lists.
        # rewards_list = tf.unstack(rewards, axis=1)
        # log_probs_list = tf.unstack(log_probs, axis=1)
        # missing = 1. - torch.cast(present, torch.float32)
        # missing_list = tf.unstack(missing, axis=1)
        missing = 1. - present.float()

        # Cumulative Discounted Returns.  The true value function V*(s).
        cumulative_rewards = []
        batch_size, seq_len = dis_predictions.size()
        for t in range(seq_len):
            cum_value = torch.zeros((batch_size, 1))
            cum_value = cum_value.cuda(self.device)
            for s in range(t, seq_len):
                cum_value_tmp = missing[:, s] * np.power(self.gamma, (s - t)) * rewards[:, s]
                cum_value_tmp = cum_value_tmp.unsqueeze(dim=1)
                cum_value += cum_value_tmp
            cumulative_rewards.append(cum_value)
        cumulative_rewards = torch.stack(cumulative_rewards, dim=1).squeeze()

        ## REINFORCE with different baselines.
        # We create a separate critic functionality for the Discriminator.  This
        # will need to operate unidirectionally and it may take in the past context.
        # Critic loss calculated from the estimated value function \hat{V}(s)
        # versus the true value function V*(s).
        critic_loss = self.create_critic_loss(cumulative_rewards, estimated_values, present)

        # Baselines are coming from the critic's estimated state values.
        # baselines = tf.unstack(estimated_values, axis=1)
        baselines = estimated_values

        ## Calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
        advantages = []
        final_gen_objective = torch.zeros([batch_size, 1]).cuda(self.device)
        for t in range(seq_len):
            log_probability = log_probs[:, t].unsqueeze(dim=1)
            cum_advantage = torch.zeros((batch_size, 1))
            cum_advantage = cum_advantage.cuda(self.device)
            for s in range(t, seq_len):
                cum_advantage_tmp = missing[:, s] * np.power(self.gamma, (s - t)) * rewards[:, s]
                cum_advantage_tmp = cum_advantage_tmp.unsqueeze(dim=1)
                cum_advantage += cum_advantage_tmp
            cum_advantage -= baselines[:, t].unsqueeze(dim=1)
            # Clip advantages.
            cum_advantage = torch.clamp(cum_advantage, -self.advantage_clipping, self.advantage_clipping)
            advantage = missing[:, t].unsqueeze(dim=1) * cum_advantage
            advantages.append(advantage)
            # cum_advantage.detach()
            final_gen_objective += torch.mul(log_probability, missing[:, t].unsqueeze(dim=1) * cum_advantage)
        final_gen_objective = -torch.sum(final_gen_objective) / batch_size  # max the reward
        maintain_averages_op = None
        advantages = torch.stack(advantages, dim=1)

        # return [
        #     final_gen_objective, log_probs, rewards, advantages, baselines,
        #     maintain_averages_op, critic_loss, cumulative_rewards
        # ]
        return final_gen_objective, critic_loss

    def generate(self, corpus):
        number_to_gen = 10
        seq_len = 20
        num_batch = number_to_gen // self.batch_size + 1 if number_to_gen != self.batch_size else 1
        samples = torch.zeros(num_batch * self.batch_size, seq_len).long()  # larger than num_samples
        inputs = torch.zeros(self.batch_size, seq_len).long().cuda(self.device)
        input_length = torch.Tensor([seq_len]*self.batch_size).float().cuda(self.device)
        targets = torch.zeros( self.batch_size, seq_len).long().cuda(self.device)
        targets_present = torch.zeros(self.batch_size, seq_len).byte().cuda(self.device)
        idx2token = corpus.idx2token

        for b in range(num_batch):
            sample, _, _ = self.forward(inputs, input_length, targets, targets_present)

            assert sample.shape == (self.batch_size, seq_len)
            samples[b * self.batch_size:(b + 1) * self.batch_size, :] = sample

        samples = samples[:number_to_gen, :]
        samples = samples.tolist()
        samples = [[idx2token[w] for w in sen] for sen in samples]

        return samples
