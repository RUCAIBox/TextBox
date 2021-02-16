# @Time   : 2021/2/3
# @Author : Zhipeng Chen
# @Email  : zhipeng_chen@ruc.edu.cn

r"""
Attr2Seq
################################################
Reference:
    Li Dong et al. "Learning to Generate Product Reviews from Attributes" in 2017.
"""

import torch
import torch.nn as nn

from textbox.model.abstract_generator import AttributeGenerator
from textbox.module.Decoder.rnn_decoder import AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling, greedy_search, Beam_Search_Hypothesis

class Attr2Seq(AttributeGenerator):
    r"""Attribute Encoder and RNN-based Decoder architecture is a basic frame work for Attr2Seq text generation.
    """
    
    def __init__(self, config, dataset):
        super(Attr2Seq, self).__init__(config, dataset)

        self.sum = 0

        # load constant of decoder
        self.rnn_type = 'lstm'
        self.attention_type = 'LuongAttention'
        self.alignment_method = 'concat'

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_dec_layers = config['num_dec_layers']
        self.dropout_ratio = config['dropout_ratio']
        self.strategy = config['decoding_strategy']

        if (self.strategy not in ['topk_sampling', 'greedy_search', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))
        if (self.strategy == 'beam_search'):
            self.beam_size = config['beam_size']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.source_token_embedder = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
        )

        if config['share_vocab']:
            self.target_token_embedder = self.source_token_embedder
        else:
            self.target_token_embedder = nn.Embedding(
                self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx
            )
        
        self.decoder = AttentionalRNNDecoder(
            self.embedding_size, self.hidden_size, self.embedding_size, self.num_dec_layers, self.rnn_type,
            self.dropout_ratio, self.attention_type, self.alignment_method
        )

        self.batch_size = config['train_batch_size']
        self.device = config['device']

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        self.max_target_length = config['target_max_seq_length']

        # parameters initialization
        self.apply(xavier_normal_initialization)

        for i in range (self.attribute_num):
            self.attribute_size[i] = self.attribute_size[i] + 1
            
        self.w = nn.ModuleList(
            [nn.Linear(self.attribute_size[i], self.embedding_size) for i in range (self.attribute_num)]
        )

        self.H = nn.Parameter(torch.rand((self.num_dec_layers * self.hidden_size, self.attribute_num * self.embedding_size), requires_grad=True)).to(self.device)
        self.b = nn.Parameter(torch.rand((self.num_dec_layers * self.hidden_size, 1), requires_grad=True)).to(self.device)
        self.c = nn.Parameter(torch.zeros((self.num_dec_layers, 1, self.hidden_size), requires_grad=True)).to(self.device)


    def encoder(self, source_idx, source_length):
        r""" 
            Args:
                source_idx (Torch.Tensor): source attribute index, shape: [source_length, attribute_size].
                source_length (integer): size of source.
            
            Returns:
                tuple:
                    - Torch.Tensor: output features, shape: [source_length, attribute_num, embeding_size].
                    - Torch.Tensor: hidden states, shape: [source_length, num_dec_layers, hidden_size].
        """
    
        # source_idx[i]: shape[source_length, attribute_size[i]].
        # w[i]: shape[attribute_size[i], embedding_size].
        # g[i]: shape[embedding_size, source_length].
        self.g = [torch.zeros((self.embedding_size, source_length), requires_grad=True).to(self.device) for i in range (self.attribute_num)]
        for i in range (self.attribute_num):
            # print("size of w[i]: ", self.w[i].size())
            # print("size of source_idx[i]: ", source_idx[i].size())
            # print("size of source_idx[i] after transpose: ", (source_idx[i].transpose(0, 1)).size())
            # self.g[i] = torch.mm(self.w[i], source_idx[i].transpose(0, 1))
            self.g[i] = self.w[i](source_idx[i]).transpose(0, 1)

        # change the shape of g[], save in g1. New shape; [source_length, attribute_num * embedding_size, 1].
        g1 = torch.zeros(source_length, self.attribute_num * self.embedding_size, 1).to(self.device)
        for i in range (source_length):
            for j in range (self.attribute_num):
                g1[i, (j * self.embedding_size) : ((j + 1) * self.embedding_size), :] = self.g[j][:, i].unsqueeze(1)
        
        #outputs (Torch.Tensor): shape: [source_length, attribute_num, embedding_size].
        outputs = g1.contiguous().view(source_length, self.attribute_num, self.embedding_size)
        
        # H  (Torch.Tensor): shape: [num_dec_layers * hidden_size, embedding_size * attribute_num].
        # H1 (Torch.Tensor): shape: [source_length, num_dec_layers * hidden_size, embedding_size * attribute_num].
        tmp = self.H.unsqueeze(0)
        H1 = tmp
        for i in range (source_length - 1):
            H1 = torch.cat((H1, tmp), 0)

        # b  (Torch.Tensor): shape: [num_dec_layers * hidden_size, 1].
        # b1 (Torch.Tensor): shape: [source_length, num_dec_layers * hidden_size, 1].
        tmp = self.b.unsqueeze(0)
        b1 = tmp
        for i in range (source_length - 1):
            b1 = torch.cat((b1, tmp), 0)

        # b1 (Torch.Tensor): shape: [source_length, num_dec_layers * hidden_size, 1].
        # g1 (Torch.Tensor): shape: [source_length, attribute_num * embedding_size, 1].
        # H1 (Torch.Tensor): shape: [source_length, num_dec_layers * hidden_size, embedding_size * attribute_num].
        # a  (Torch.Tensor): shape: [source_length, num_dec_layers * hidden_size, 1].
        a = torch.tanh(torch.bmm(H1, g1) + b1)

        # hidden_states (Torch.Tensor): shape: [num_dec_layers, source_length, hidden_size].
        hidden_states = a.contiguous().view(source_length, self.num_dec_layers, self.hidden_size)
        hidden_states = hidden_states.transpose(0, 1)

        return outputs, hidden_states

    def generate(self, eval_dataloader):
        print('in generator.')

        generate_corpus = []
        idx2token = eval_dataloader.idx2token

        num = 0

        for batch_data in eval_dataloader:
            source_idx = batch_data['attribute_idx']
            source_size = source_idx.size(0)

            num += 1
            print('source_size: ', source_size, ';', 'num: ', num)

            source_embeddings = [torch.zeros(source_size, self.attribute_size[i]).to(self.device) for i in range(self.attribute_num)]
            for i in range (self.attribute_num):
                for id in range (source_size):
                    source_embeddings[i][id, source_idx[id][i]] = 1
            
            encoder_outputs, encoder_states = self.encoder(source_embeddings, source_size)
            encoder_masks = torch.ne(source_idx, self.padding_token_idx)
            
            for bid in range(source_size):
                decoder_states = (encoder_states[:, bid, :].unsqueeze(1), self.c)
                # decoder_states = encoder_states[:, bid, :].unsqueeze(1)
                encoder_output = encoder_outputs[bid, :, :].unsqueeze(0)
                encoder_mask = encoder_masks[bid, :].unsqueeze(0)
                generate_tokens = []
                input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)

                if (self.strategy == 'beam_search'):
                    hypothesis = Beam_Search_Hypothesis(
                        self.beam_size, self.sos_token_idx, self.eos_token_idx, self.device, idx2token
                    )

                self.max_target_length = 200
                for gen_idx in range(self.max_target_length):
                    decoder_input = self.target_token_embedder(input_seq)
                    decoder_outputs, decoder_states, _ = self.decoder(
                        decoder_input, decoder_states, encoder_output, encoder_mask
                    )

                    token_logits = self.vocab_linear(decoder_outputs)
                    if (self.strategy == 'topk_sampling'):
                        token_idx = topk_sampling(token_logits).item()
                    elif (self.strategy == 'greedy_search'):
                        token_idx = greedy_search(token_logits).item()
                    elif (self.strategy == 'beam_search'):
                        input_seq, tmp_decoder_states, encoder_output, encoder_mask = \
                            hypothesis.step(gen_idx, token_logits, decoder_states, encoder_output, encoder_mask)
                        # decoder_states = (tmp_decoder_states, decoder_states[1])

                    if (self.strategy in ['topk_sampling', 'greedy_search']):
                        if token_idx == self.eos_token_idx:
                            break
                        else:
                            generate_tokens.append(idx2token[token_idx])
                            input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                    elif (self.strategy == 'beam_search'):
                        if (hypothesis.stop()):
                            break

                if (self.strategy == 'beam_search'):
                    generate_tokens = hypothesis.generate()

                generate_corpus.append(generate_tokens)

        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        self.sum = self.sum + 1
        print('in calculate_loss: ', self.sum, '.')

        target_length = corpus['target_length']
        # attribute_idx (Torch.Tensor): shape: [source_length, attribute_num].
        attribute_idx = corpus['attribute_idx']
        source_length = attribute_idx.size(0)
        # target_idx (torch.Tensor): shape: [source_length, target_length].
        target_idx = corpus['target_idx']

        print ('batch_size: ', self.batch_size, '; ', 'source_length: ', source_length, '; ', 'attribute_num: ', self.attribute_num, '; ', 'attribute_size: ', self.attribute_size[0], ', ', self.attribute_size[1], ', ',self.attribute_size[2])
        #source_idx[i] (Torch.Tensor): shape: [source_length, attribute_size[i]]
        source_idx = [torch.zeros(source_length, self.attribute_size[i]).to(self.device) for i in range(self.attribute_num)]
        for i in range (self.attribute_num):
            for id in range (source_length):
                if (attribute_idx[id][i] >= self.attribute_size[i]):
                    print ('attribute_idx error.')
                source_idx[i][id, attribute_idx[id][i]] = 1

        encoder_outputs, encoder_states = self.encoder(source_idx, source_length)

        input_text = target_idx[:, :-1]
        target_text = target_idx[:, 1:]
        input_embeddings = self.dropout(self.target_token_embedder(input_text))
        
        # print("size of input_embeddings: ", input_embeddings.size())
        # print("size of encoder_states: ", encoder_states.size())
        # print("size of encoder_outputs: ", encoder_outputs.size())
        c1 = self.c
        for i in range (source_length - 1):
            c1 = torch.cat((c1, self.c), 1)
        encoder_masks = torch.ne(attribute_idx, self.padding_token_idx)
        decoder_outputs, decoder_states, _ = self.decoder(input_embeddings, (encoder_states.contiguous(), c1), encoder_outputs, encoder_masks)

        # token_logits (Torch.Tensor): shape: [source_length, target_length, vocabulary_size].
        token_logits = self.vocab_linear(decoder_outputs)

        # token_logits.view(-1, token_logits.size(-1)) (Torch.Tensor): shape: [source_length * target_length, vocabulary_size].
        # target_text.contiguous().view(-1) (Torch.Tensor): shape: [source_length * target_length].
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        loss = loss.sum(dim = 1) / (target_length - 1).float()
        loss = loss.mean()
        return loss
