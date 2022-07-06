# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/26
# @Author : Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn

r"""
RNN Decoder
###############
"""

import torch
from torch import nn
import torch.nn.functional as F
from textbox.module.Attention.attention_mechanism import LuongAttention, BahdanauAttention, MonotonicAttention


class BasicRNNDecoder(torch.nn.Module):
    r"""
    Basic Recurrent Neural Network (RNN) decoder.
    """

    def __init__(self, embedding_size, hidden_size, num_dec_layers, rnn_type, dropout_ratio=0.0):
        super(BasicRNNDecoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_dec_layers = num_dec_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == "gru":
            self.decoder = nn.GRU(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == "rnn":
            self.decoder = nn.RNN(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("The RNN type in decoder must in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, hidden_states=None):
        r""" Implement the decoding process.

        Args:
            input_embeddings (Torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [num_layers * num_directions, batch_size, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        if not isinstance(hidden_states, tuple):
            hidden_states = hidden_states.contiguous()
        outputs, hidden_states = self.decoder(input_embeddings, hidden_states)
        return outputs, hidden_states


class AttentionalRNNDecoder(torch.nn.Module):
    r"""
    Attention-based Recurrent Neural Network (RNN) decoder.
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        context_size,
        num_dec_layers,
        rnn_type,
        dropout_ratio=0.0,
        attention_type='LuongAttention',
        alignment_method='concat'
    ):
        super(AttentionalRNNDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_type = rnn_type
        self.attention_type = attention_type
        self.alignment_method = alignment_method

        if attention_type == 'LuongAttention':
            self.attentioner = LuongAttention(self.context_size, self.hidden_size, self.alignment_method)
            dec_input_size = embedding_size
        elif attention_type == 'BahdanauAttention':
            self.attentioner = BahdanauAttention(self.context_size, self.hidden_size)
            dec_input_size = embedding_size + context_size
        elif attention_type == 'MonotonicAttention':
            self.attentioner = MonotonicAttention(self.context_size, self.hidden_size)
            dec_input_size = embedding_size
        else:
            raise ValueError("Attention type must be in ['LuongAttention', 'BahdanauAttention', 'MonotonicAttention'].")

        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'gru':
            self.decoder = nn.GRU(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'rnn':
            self.decoder = nn.RNN(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("RNN type in attentional decoder must be in ['lstm', 'gru', 'rnn'].")

        self.attention_dense = nn.Linear(hidden_size + context_size, hidden_size)

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_type))

    def forward(
        self, input_embeddings, hidden_states=None, encoder_outputs=None, encoder_masks=None, previous_probs=None
    ):
        r""" Implement the attention-based decoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.
            encoder_outputs (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_masks (Torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        if encoder_outputs is not None and encoder_masks is None:
            encoder_masks = torch.ones(encoder_outputs.size(0), encoder_outputs.size(1)).to(encoder_outputs.device)

        decode_length = input_embeddings.size(1)

        probs = previous_probs
        all_outputs = []
        for step in range(decode_length):

            if self.attention_type == 'BahdanauAttention':
                # only top layer
                if self.rnn_type == 'lstm':
                    hidden = hidden_states[0][-1]
                else:
                    hidden = hidden_states[-1]
                context, probs = self.attentioner(hidden, encoder_outputs, encoder_masks)
                embed = input_embeddings[:, step, :].unsqueeze(1)
                inputs = torch.cat((embed, context), dim=-1)
            else:
                inputs = input_embeddings[:, step, :].unsqueeze(1)
                context = None

            if (not isinstance(hidden_states, tuple)):
                hidden_states = hidden_states.contiguous()
            outputs, hidden_states = self.decoder(inputs, hidden_states)

            if self.attention_type == 'LuongAttention' and context is None:
                context, probs = self.attentioner(outputs, encoder_outputs, encoder_masks)
            elif self.attention_type == 'MonotonicAttention' and context is None:
                if self.training:
                    context, probs = self.attentioner.soft(outputs, encoder_outputs, encoder_masks, probs)
                else:
                    context, probs = self.attentioner.hard(outputs, encoder_outputs, encoder_masks, probs)
            elif self.attention_type == 'BahdanauAttention':
                pass
            else:
                raise NotImplementedError("No such attention type {} for decoder output.".format(self.attention_type))
            outputs = self.attention_dense(torch.cat((outputs, context), dim=2))
            all_outputs.append(outputs)

        outputs = torch.cat(all_outputs, dim=1)
        return outputs, hidden_states, probs


class PointerRNNDecoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_size,
            context_size,
            num_dec_layers,
            rnn_type,
            dropout_ratio=0.0,
            is_attention=False,
            is_pgen=False,
            is_coverage=False
    ):
        super(PointerRNNDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_type = rnn_type

        dec_input_size = embedding_size
        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'gru':
            self.decoder = nn.GRU(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_type == 'rnn':
            self.decoder = nn.RNN(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("RNN type in attentional decoder must be in ['lstm', 'gru', 'rnn'].")

        self.is_attention = is_attention
        self.is_pgen = is_pgen and is_attention
        self.is_coverage = is_coverage and is_attention
        self.vocab_linear = nn.Linear(hidden_size, vocab_size)

        if self.is_attention:
            self.x_context = nn.Linear(embedding_size + context_size, embedding_size)
            self.attention = LuongAttention(self.context_size, self.hidden_size, 'concat', self.is_coverage)
            self.attention_dense = nn.Linear(hidden_size + context_size, hidden_size)

        if self.is_pgen:
            self.p_gen_linear = nn.Linear(context_size + hidden_size + embedding_size, 1)

    def forward(self, input_embeddings, decoder_hidden_states, kwargs=None):
        if not self.is_attention:
            decoder_outputs, decoder_hidden_states = self.decoder(input_embeddings, decoder_hidden_states)
            vocab_dists = F.softmax(self.vocab_linear(decoder_outputs), dim=-1)
            return vocab_dists, decoder_hidden_states, kwargs

        else:
            vocab_dists = []
            encoder_outputs = kwargs['encoder_outputs']  # B x src_len x 256
            encoder_masks = kwargs['encoder_masks']  # B x src_len
            context = kwargs['context']  # B x 1 x 256

            extra_zeros = None
            source_extended_idx = None
            if self.is_pgen:
                extra_zeros = kwargs['extra_zeros']  # B x max_oovs_num
                source_extended_idx = kwargs['source_extended_idx']  # B x src_len (contains oovs ids)

            coverage = None
            attn_dists = None
            coverages = None
            if self.is_coverage:
                coverage = kwargs['coverages']
                coverages = []
                attn_dists = []

            tgt_len = input_embeddings.size(1)

            for step in range(tgt_len):
                step_input_embeddings = input_embeddings[:, step, :].unsqueeze(1)  # B x 1 x 128

                x = self.x_context(torch.cat((step_input_embeddings, context), dim=-1))  # B x 1 x 128

                decoder_outputs, decoder_hidden_states = self.decoder(x, decoder_hidden_states)  # B x 1 x 256

                context, attn_dist, coverage = self.attention(decoder_outputs, encoder_outputs,
                                                              encoder_masks, coverage)  # B x 1 x src_len

                vocab_logits = self.vocab_linear(self.attention_dense(torch.cat((decoder_outputs, context), dim=-1)))
                vocab_dist = F.softmax(vocab_logits, dim=-1)  # B x 1 x vocab_size

                if self.is_pgen:
                    p_gen_input = torch.cat((context, decoder_outputs, x), dim=-1)  # B x 1 x (256 + 256 + 128)
                    p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # B x 1 x 1
                    copy_attn_dist = (1 - p_gen) * attn_dist  # B x 1 x src_len

                    # B x 1 x (vocab_size+max_oovs_num)
                    extended_vocab_dist = torch.cat(((vocab_dist * p_gen), extra_zeros.unsqueeze(1)), dim=-1)
                    # add copy probs to vocab dist
                    vocab_dist = extended_vocab_dist.scatter_add(2, source_extended_idx.unsqueeze(1), copy_attn_dist)

                if self.is_coverage:
                    attn_dists.append(attn_dist)
                    coverages.append(coverage)

                vocab_dists.append(vocab_dist)

            vocab_dists = torch.cat(vocab_dists, dim=1)  # B x tgt_len x vocab_size+(max_oovs_num)

            kwargs['context'] = context

            if self.is_coverage:
                coverages = torch.cat(coverages, dim=1)  # B x tgt_len x src_len
                attn_dists = torch.cat(attn_dists, dim=1)  # B x tgt_len x src_len
                kwargs['attn_dists'] = attn_dists
                kwargs['coverages'] = coverages

            return vocab_dists, decoder_hidden_states, kwargs
