import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class CVAEDecoder(torch.nn.Module):
    def __init__(self,
                 decoder_kernel_size,
                 input_size,
                 hidden_size,
                 drop_rate):
        super(CVAEDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        if isinstance(decoder_kernel_size, int):
            self.decoder_kernel_size = [decoder_kernel_size]
        elif isinstance(decoder_kernel_size, list):
            self.decoder_kernel_size = decoder_kernel_size
        else:
            raise NotImplementedError("Unrecognized hyper parameters: {}".format(decoder_kernel_size))
        self.decoder_drop = nn.Dropout(p=drop_rate)
        self.module_def()

        # self.decoder_dilations = [1, 2, 4]
        # self.decoder_kernels = [(400, self.hidden_size + self.input_size, 3),
        #                         (450, 400, 3),
        #                         (500, 450, 3)]
        # self.decoder_num_layers = len(self.decoder_kernels)
        # self.decoder_paddings = [self.effective_k(w, self.decoder_dilations[i]) - 1
        #                          for i, (_, _, w) in enumerate(self.decoder_kernels)]
        # self.cnn_list = [torch.nn.Conv1d(in_channels=,
        #                                  out_channels=,
        #                                  kernel_size=kernel,
        #                                  padding=self.decoder_paddings[i],
        #                                  dilation=) for i, kernel in enumerate(self.decoder_kernels)]

    def module_def(self):
        self.decoder_dilations = [1, 2, 4]
        assert len(self.decoder_kernel_size) <= 3
        # [400, 450, 500]
        decoder_kernels = []
        for i, out_channels in enumerate(self.decoder_kernel_size):
            if i == 0:
                in_channels = self.hidden_size + self.input_size
            else:
                in_channels = self.decoder_kernel_size[i-1]
            decoder_kernels.append((out_channels, in_channels, 3))

        # decoder_kernels = [(self.decoder_kernel_size[0], self.hidden_size + self.input_size, 3),
        #                    (self.decoder_kernel_size[1], self.decoder_kernel_size[0], 3),
        #                    (self.decoder_kernel_size[2], self.decoder_kernel_size[1], 3)]
        # decoder_num_layers = len(decoder_kernels)
        self.decoder_paddings = [self.effective_k(w, self.decoder_dilations[i]) - 1
                                 for i, (_, _, w) in enumerate(decoder_kernels)]
        # self.kernels = nn.ModuleList([Parameter(torch.Tensor(out_chan, in_chan, width).normal_(0, 0.05))
        #                               for out_chan, in_chan, width in decoder_kernels])
        # self._add_to_parameters(self.kernels, 'decoder_kernel')

        # self.biases = nn.Module_list([Parameter(torch.Tensor(out_chan).normal_(0, 0.05))
        #                               for out_chan, in_chan, width in decoder_kernels])

        # self._add_to_parameters(self.biases, 'decoder_bias')
        cnn_list = []
        for layer, (out_chan, in_chan, width) in enumerate(decoder_kernels):
            cnn_list.append(torch.nn.Conv1d(in_channels=in_chan,
                                            out_channels=out_chan,
                                            kernel_size=width,
                                            dilation=self.decoder_dilations[layer],
                                            padding=self.decoder_paddings[layer])
                            )

        self.cnn_list = nn.ModuleList(cnn_list)

        self.out_size = decoder_kernels[-1][0]

    @staticmethod
    def effective_k(k, d):
        """
        :param k: kernel width
        :param d: dilation size
        :return: effective kernel width when dilation is performed
        """
        return (k - 1) * d + 1

    def forward(self, decoder_input, z):
        '''

        :param decoder_input: [batch_size, length, embedding_size]
        :param z: [batch_size, hidden_size]
        :return:
        '''
        batch_size, seq_len, _ = decoder_input.size()

        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.hidden_size)
        decoder_input = torch.cat([decoder_input, z], 2)
        decoder_input = self.decoder_drop(decoder_input)

        # x is tensor with shape [batch_size, input_size=in_channels, seq_len=input_width]
        x = decoder_input.transpose(1, 2).contiguous()

        # for layer, kernel in enumerate(self.kernels):
        #     # apply conv layer with non-linearity and drop last elements of sequence to perfrom input shifting
        #     x = F.conv1d(x, kernel,
        #                  bias=self.biases[layer],
        #                  dilation=self.decoder_dilations[layer],
        #                  padding=self.decoder_paddings[layer])
        #
        #     x_width = x.size()[2]
        #     x = x[:, :, :(x_width - self.decoder_paddings[layer])].contiguous()
        #
        #     x = F.relu(x)

        for layer, cnn in enumerate(self.cnn_list):
            # apply conv layer with non-linearity and drop last elements of sequence to perfrom input shifting
            x = cnn(x)
            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.decoder_paddings[layer])].contiguous()

            x = F.relu(x)

        result = x.transpose(1, 2).contiguous()
        # (batch_size, seq_len, out_size)
        return result


class HybridDecoder(nn.Module):
    '''
    Code Reference: https://github.com/kefirski/hybrid_rvae
    '''
    def __init__(self, vocab_size, hidden_size, num_layers, embedding_size):
        super(HybridDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        
        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_size, 512, 4, 2, 0),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.ConvTranspose1d(512, 512, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.ConvTranspose1d(512, 256, 4, 2, 0),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.ConvTranspose1d(256, 256, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.ConvTranspose1d(256, 128, 4, 2, 0),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.ConvTranspose1d(128, self.vocab_size, 4, 2, 0)
        )

        self.rnn = nn.GRU(input_size=self.vocab_size + self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.hidden_to_vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, decoder_input, latent_variable):
        """
        :param latent_variable: An float tensor with shape of [batch_size, hidden_size]
        :param decoder_input: An float tensot with shape of [batch_size, max_seq_len, embed_size]
        :return: two tensors with shape of [batch_size, max_seq_len, vocab_size]
                    for estimating likelihood for whole model and for auxiliary target respectively
        """

        aux_logits = self.conv_decoder(latent_variable)
        batch_size, seq_len, _ = decoder_input.size()

        aux_logits = aux_logits[:, :seq_len, :].contiguous()

        logits, _ = self.rnn_decoder(aux_logits, decoder_input, initial_state=None)

        return logits, aux_logits

    def conv_decoder(self, latent_variable):
        latent_variable = latent_variable.unsqueeze(2)

        out = self.cnn(latent_variable)
        return torch.transpose(out, 1, 2).contiguous()

    def init_hidden(self, decoder_input):
        batch_size = decoder_input.size(0)
        device = decoder_input.device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def rnn_decoder(self, cnn_out, decoder_input, initial_state=None):
        # print(cnn_out.size(), decoder_input.size())
        logits, final_state = self.rnn(torch.cat([cnn_out, decoder_input], 2), initial_state)

        [batch_size, seq_len, _] = logits.size()
        logits = logits.contiguous().view(-1, self.hidden_size)

        logits = self.hidden_to_vocab(logits)

        logits = logits.view(batch_size, seq_len, self.vocab_size)

        return logits, final_state
