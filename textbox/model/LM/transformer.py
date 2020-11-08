# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.utils import InputType
from textbox.model.abstract_generator import UnconditionalGenerator
from textbox.module.Decoder.transformer_decoder import TransformerDecoder
from textbox.module.Attention.multi_head_attention import SinusoidalPositionalEmbedding, SelfAttentionMask
from textbox.model.init import xavier_normal_initialization


class Transformer(UnconditionalGenerator):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.NOISE

    def __init__(self, config, dataset):
        super(Transformer, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.ffn_size = config['ffn_size']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        self.position_embedder = SinusoidalPositionalEmbedding(self.embedding_size)
        self.self_attn_mask = SelfAttentionMask()

        self.decoder = TransformerDecoder(self.embedding_size, self.ffn_size, self.num_layers, self.num_heads)
        self.vocab_linear = nn.Linear(self.embedding_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def generate(self, eval_data):
        generate_corpus = []
        number_to_gen = 10
        idx2token = eval_data.idx2token
        for _ in range(number_to_gen):
            generate_token_idx = [self.sos_token_idx]
            generate_tokens = []
            for gen_idx in range(100):
                input_seq = torch.LongTensor([generate_token_idx]).to(self.device)
                input_embedding = self.token_embedder(input_seq) + self.position_embedder(input_seq).to(self.device)
                self_padding_mask = torch.eq(input_seq, self.padding_token_idx).to(self.device)
                self_attn_mask = self.self_attn_mask(input_seq.size(-1)).bool().to(self.device)

                token_logits = self.decoder(input_embedding,
                                            self_padding_mask=self_padding_mask,
                                            self_attn_mask=self_attn_mask)
                token_logits = token_logits[:, -1, :]

                topv, topi = torch.log(F.softmax(token_logits, dim=-1) + 1e-12).data.topk(k=4)
                topi = topi.squeeze()
                token_idx = topi[0].item()
                if token_idx == self.eos_token_idx or gen_idx >= 100:
                    break
                else:
                    generate_token_idx.append(token_idx)
                    generate_tokens.append(idx2token[token_idx])
            generate_corpus.append(generate_tokens)
        return generate_corpus

    def calculate_loss(self, corpus):
        input_text = corpus['target_text'][:, :-1]
        target_text = corpus['target_text'][:, 1:]

        input_embeddings = self.token_embedder(input_text) + self.position_embedder(input_text).to(self.device)
        self_padding_mask = torch.eq(input_text, self.padding_token_idx).to(self.device)
        self_attn_mask = self.self_attn_mask(input_text.size(-1)).bool().to(self.device)

        token_repre = self.decoder(input_embeddings,
                                   self_padding_mask=self_padding_mask,
                                   self_attn_mask=self_attn_mask)

        token_logits = self.vocab_linear(token_repre)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        target_text = target_text.contiguous().view(-1)

        loss = self.loss(token_logits, target_text)
        return loss
