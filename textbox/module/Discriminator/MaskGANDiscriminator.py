# @Time   : 2020/12/2
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization


class MaskGANDiscriminator(GenerativeAdversarialNet):
    r"""RNN-based Encoder-Decoder architecture is a basic framework for conditional text generation.

    """

    # input_type = InputType.PAIRTEXT

    def __init__(self, config, dataset):
        super(MaskGANDiscriminator, self).__init__(config, dataset)

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

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
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
        self.fc_linear = nn.Linear(self.hidden_size, 1)
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
        zeroth_input_present = torch.ones_like(targets_present)[:, 1].unsqueeze(dim=-1)  # bs*1

        # 将第0个输入和target_present表示的剩余输入一起拼接成bs*seq_len的mask矩阵
        inputs_present = torch.cat([zeroth_input_present, targets_present], dim=1)[:, :-1]  # bs*seq_len

        # 使用where函数根据inputs_present确定是否maskinputs
        masked_input = torch.where(inputs_present, inputs, inputs_missing)

        return masked_input

    def forward(self, inputs, inputs_length, sequence, targets_present, is_training=True):
        r"""输入真实的补齐的句子以及句子长度以及target_mask矩阵，通过seq2seq输出预测生成的target输出

        Args:
            inputs: 真实句子 bs*seq_len
            inputs_length:  句子长度 list[bs]
            targets_present: 目标句子的mask矩阵 bs*seq_len
            sequence: 真实的目标句子或Generator生成的伪句子
            is_training: 控制训练还是验证
        Returns:
            prediction: 预测是真实的token的概率
        """
        # 得到mask_input
        masked_inputs = self.mask_input(inputs, targets_present)

        # 将input转换为embedding
        sequence_rnn_inputs = self.target_token_embedder(sequence)
        masked_rnn_inputs = self.source_token_embedder(masked_inputs)

        # 将masked_rnn_input传入encoder得到encoder_final_state和encoder_outputs
        encoder_outputs, encoder_final_state = self.encoder(masked_rnn_inputs, inputs_length)

        # 使用decoder依次传入masked_input解码生成预测概率，注意每个时刻的输入都使用sequence
        seq_len = inputs.size()[1]
        hidden_size = encoder_final_state
        predictions = []
        for t in range(seq_len):
            input = sequence_rnn_inputs[:, t].unsqueeze(dim=1)  # bs*1*emb_dim
            # rnn_output, hidden_size = self.decoder(input, hidden_size, encoder_outputs)
            rnn_output, hidden_size = self.decoder(input, hidden_size)  # need fix attention decoder bs*1*hid_dim
            prediction = self.fc_linear(rnn_output)  # bs*1*1
            prediction = prediction.squeeze(dim=1)  # bs*1
            predictions.append(prediction)  # bs*1
        # 收集decoder解码过程中的prediction
        predictions = torch.stack(predictions, dim=1)  # bs*seq_len*1
        predictions = predictions.squeeze(dim=2)  # bs*seq_len
        return predictions

    def critic(self, fake_sequence):
        r"""Define the Critic graph which is derived from the seq2seq Discriminator. This will be
        initialized with the same parameters as the language model and will share the forward RNN
        components with the Discriminator. This estimates the V(s_t), where the state s_t = x_0,...,x_t-1.

        Args:
            fake_sequence: 生成的句子 bs*seq_len

        Returns:
            values: 价值得分 bs*seq_len
        """
        # 依次遍历每一个时刻，直接传入discriminator的decoder端得到输出即为该时刻的价值函数值
        # 注意传入的sequence比fake_sequence要左移一个时刻，最开始用全0作为输入
        fake_sequence = self.target_token_embedder(fake_sequence)
        seq_len = fake_sequence.size()[1]
        values = []
        hidden_size = None
        for t in range(seq_len):
            if t == 0:
                input = torch.zeros_like(fake_sequence[:, 0].unsqueeze(dim=1))
            else:
                input = fake_sequence[:, t - 1].unsqueeze(dim=1)
            rnn_output, hidden_size = self.decoder(input, hidden_size)
            value = self.fc_linear(rnn_output)  # bs*1*1
            values.append(value)
        # 压缩values的形状返回
        values = torch.stack(values, dim=1).squeeze()  # bs*seq_len
        return values

    def calculate_dis_loss(self, fake_prediction, real_prediction, target_present):
        r"""Compute Discriminator loss across real/fake

        Args:
            fake_prediction:
            real_prediction:
            target_present:

        Returns:

        """
        # 将target_present转换为missing矩阵：0代表不计算，1代表计算
        missing = 1 - target_present.float()

        # 得到real_label
        batch_size, seq_len = real_prediction.size()
        real_label = torch.ones_like(real_prediction, dtype=torch.float)  # bs*seq_len
        fake_label = torch.zeros_like(fake_prediction, dtype=torch.float)
        # 分别计算sigmoid_cross_entropy
        real_prediction_sigmoid = torch.sigmoid(real_prediction)  # bs*seq_len
        fake_prediction_sigmoid = torch.sigmoid(fake_prediction)  # bs*seq_len
        loss = nn.BCELoss(weight=missing)
        real_loss = loss(real_prediction_sigmoid, real_label)
        fake_loss = loss(fake_prediction_sigmoid, fake_label)
        loss = (real_loss + fake_loss) / 2
        # 返回loss
        return loss

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

    def calculate_loss(self, real_sequence, lengths, fake_sequence, targets_present):
        fake_prediction = self.forward(real_sequence, lengths, fake_sequence, targets_present)
        real_prediction = self.forward(real_sequence, lengths, real_sequence, targets_present)
        return self.calculate_dis_loss(fake_prediction, real_prediction, targets_present)
