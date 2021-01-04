# @Time   : 2020/12/26
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn

r"""
MaskGAN Discriminator
#####################
"""

import torch
import torch.nn as nn

from textbox.model.abstract_generator import GenerativeAdversarialNet
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization


class MaskGANDiscriminator(GenerativeAdversarialNet):
    r""" RNN-based Encoder-Decoder architecture for MaskGAN discriminator
    """

    def __init__(self, config, dataset):
        super(MaskGANDiscriminator, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.bidirectional = config['bidirectional']
        self.alignment_method = config['alignment_method']
        self.dropout_ratio = config['dropout_ratio']
        self.attention_type = config['attention_type']
        self.context_size = config['context_size']
        self.attention_type = config['attention_type']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx
        self.mask_token_idx = dataset.user_token_idx[0]

        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type,
                                       self.dropout_ratio, self.bidirectional)

        if self.attention_type is not None:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.hidden_size, self.context_size,
                                                 self.num_dec_layers, self.rnn_type, self.dropout_ratio,
                                                 self.attention_type, self.alignment_method)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.hidden_size, self.num_dec_layers,
                                           self.rnn_type, self.dropout_ratio)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.fc_linear = nn.Linear(self.hidden_size, 1)
        self.critic_fc_linear = nn.Linear(self.hidden_size, 1)

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

    def forward(self, inputs, inputs_length, sequence, targets_present, embedder):
        r"""Predict the real prob of the filled_in token using real sentence and fake sentence

        Args:
            inputs: real input bs*seq_len
            inputs_length:  sentences length list[bs]
            sequence: real target or the generated sentence by Generator
            targets_present: target sentences present matrix bs*seq_len
            embedder: shared embedding with generator

        Returns:
            prediction: the real prob of filled_in token predicted by discriminator
        """

        masked_inputs = self.mask_input(inputs, targets_present)

        sequence_rnn_inputs = embedder(sequence)  # bs*seq_len*emb
        masked_rnn_inputs = embedder(masked_inputs)
        sequence_rnn_inputs.detach()
        masked_rnn_inputs.detach()

        if self.rnn_type == "lstm":
            encoder_outputs, encoder_final_state = self.encoder(masked_rnn_inputs, inputs_length)
            h, c = encoder_final_state
            h = h.contiguous()
            c = c.contiguous()
            encoder_final_state = (h, c)
        else:
            encoder_outputs, encoder_final_state = self.encoder(masked_rnn_inputs, inputs_length)

        seq_len = inputs.size()[1]
        hidden_size = encoder_final_state
        predictions = []
        values = []
        for t in range(seq_len):
            input = sequence_rnn_inputs[:, t, :].unsqueeze(dim=1)  # bs*1*emb_dim
            if self.attention_type is not None:
                encoder_mask = torch.ones_like(inputs)
                rnn_output, hidden_size, _ = self.decoder(input, hidden_size, encoder_outputs, encoder_mask)
            else:
                rnn_output, hidden_size = self.decoder(input, hidden_size)
            prediction = self.fc_linear(rnn_output)  # bs*1*1
            prediction = prediction.squeeze(dim=2).squeeze(dim=1)  # bs
            predictions.append(prediction)  # bs
            # value = self.critic_fc_linear(rnn_output) # bs*1*1
            # value = value.squeeze(dim=2).squeeze(dim=1)  # bs
            # values.append(value)

        predictions = torch.stack(predictions, dim=1)  # bs*seq_len
        # values = torch.stack(values, dim=1)  # bs*seq_len
        return predictions, values

    def critic(self, fake_sequence, embedder):
        r"""Define the Critic graph which is derived from the seq2seq Discriminator. This will be
        initialized with the same parameters as the language model and will share the forward RNN
        components with the Discriminator. This estimates the V(s_t), where the state s_t = x_0,...,x_t-1.

        Args:
            fake_sequence: sequence generated bs*seq_len

        Returns:
            values: bs*seq_len
        """
        fake_sequence = embedder(fake_sequence)
        fake_sequence = fake_sequence.detach()

        seq_len = fake_sequence.size()[1]
        values = []
        hidden_size = None
        for t in range(seq_len):
            if t == 0:
                input = torch.zeros_like(fake_sequence[:, 0].unsqueeze(dim=1))
            else:
                input = fake_sequence[:, t - 1].unsqueeze(dim=1)
            rnn_output, hidden_size = self.decoder.decoder(input, hidden_size)
            rnn_output_ = rnn_output.detach()  # only update critic_linear
            value = self.critic_fc_linear(rnn_output_)  # bs*1*1
            values.append(value)

        values = torch.stack(values, dim=1).squeeze()  # bs*seq_len
        return values

    def calculate_dis_loss(self, fake_prediction, real_prediction, target_present):
        r"""Compute Discriminator loss across real/fake
        """
        missing = 1 - target_present.float()  # bs*seq_len

        real_label = torch.ones_like(real_prediction, dtype=torch.float)  # bs*seq_len
        fake_label = torch.zeros_like(fake_prediction, dtype=torch.float)

        loss = nn.BCEWithLogitsLoss(weight=missing, reduction='none')
        real_prediction = torch.clamp(real_prediction, min=-5, max=5)
        fake_prediction = torch.clamp(fake_prediction, min=-5, max=5)
        real_loss = loss(real_prediction, real_label)
        real_loss = torch.sum(real_loss) / torch.sum(missing)

        fake_loss = loss(fake_prediction, fake_label)
        fake_loss = torch.sum(fake_loss) / torch.sum(missing)
        loss = (real_loss + fake_loss) / 2

        return loss

    def create_critic_loss(self, cumulative_rewards, estimated_values, target_present):
        r"""Compute Critic loss in estimating the value function.  This should be an
        estimate only for the missing elements.
        """
        missing = 1 - target_present.float()
        estimated_values = torch.where(target_present, cumulative_rewards, estimated_values)
        loss = nn.MSELoss(reduction='none')
        l = loss(estimated_values, cumulative_rewards)
        l = torch.sum(l) / torch.sum(missing)
        return l

    def calculate_loss(self, real_sequence, lengths, fake_sequence, targets_present, embedder):
        r"""Calculate discriminator loss
        """
        fake_prediction, _ = self.forward(real_sequence, lengths, fake_sequence, targets_present, embedder)
        real_prediction, _ = self.forward(real_sequence, lengths, real_sequence, targets_present, embedder)

        return self.calculate_dis_loss(fake_prediction, real_prediction, targets_present)

    def mask_target_present(self, targets_present, lengths):
        batch_size, seq_len = targets_present.size()
        for i in range(batch_size):
            len = lengths[i].int()
            targets_present[i, len:] = True
        return targets_present
