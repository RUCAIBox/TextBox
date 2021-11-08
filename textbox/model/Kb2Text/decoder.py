# @Time   : 2021/9/22
# @Author : ZiKang Liu
# @Email  : jason8121@foxmail.com

import torch
from torch import nn
from textbox.module.Attention.attention_mechanism import MultiHeadAttention, LuongAttention


class Decoder(torch.nn.Module):
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
            alignment_method='concat',
            num_heads=4,
            attn_weight_dropout_ratio=0.1
    ):
        super(Decoder, self).__init__()

        self.attn_weight_dropout_ratio = attn_weight_dropout_ratio
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_type = rnn_type
        self.attention_type = attention_type
        self.alignment_method = alignment_method

        dec_input_size = embedding_size
        self.decoder = nn.LSTM(dec_input_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        self.attention_dense = nn.Linear(hidden_size + 2 * context_size, hidden_size)
        self.attentioner = LuongAttention(self.context_size, self.hidden_size, self.alignment_method)
        self.item_attentioner = MultiHeadAttention(embedding_size, self.num_heads, self.attn_weight_dropout_ratio)
        self.title_attentioner = MultiHeadAttention(embedding_size, self.num_heads, self.attn_weight_dropout_ratio)

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape:
            [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        h_0 = torch.zeros(self.num_dec_layers, batch_size, 2 * self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_dec_layers, batch_size, 2 * self.hidden_size).to(device)
        hidden_states = (h_0, c_0)
        return hidden_states

    def forward(
            self, input_embeddings, hidden_states=None, encoder_outputs_items=None, encoder_outputs_titles=None,
            encoder_item_masks=None, encoder_title_masks=None
    ):
        r""" Implement the attention-based decoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.
            encoder_outputs_items: (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_outputs_titles: (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_item_masks (Torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.
            encoder_title_masks (Torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        decode_length = input_embeddings.size(1)

        all_outputs = []
        for step in range(decode_length):
            inputs = input_embeddings[:, step, :].unsqueeze(1)
            outputs, hidden_states = self.decoder(inputs, hidden_states)

            h_ = hidden_states[0].transpose(0, 1)
            context_items = h_ + self.item_attentioner(
                h_, encoder_outputs_items, encoder_outputs_items, encoder_item_masks)[0]
            context_titles = h_ + self.title_attentioner(
                h_, encoder_outputs_titles, encoder_outputs_titles, encoder_title_masks)[0]
            context = torch.cat((context_items, context_titles), dim=2).squeeze(1)
            # outputs = self.attention_dense(torch.cat((outputs, context), dim=1))
            outputs = torch.cat((outputs.squeeze(1), context), dim=1)
            all_outputs.append(outputs)

        outputs = torch.stack(all_outputs, dim=1)
        return outputs, hidden_states
