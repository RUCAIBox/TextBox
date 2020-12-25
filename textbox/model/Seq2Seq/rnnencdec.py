# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.abstract_generator import ConditionalGenerator
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling


class RNNEncDec(ConditionalGenerator):
    r"""RNN-based Encoder-Decoder architecture is a basic framework for conditional text generation.

    """

    def __init__(self, config, dataset):
        super(RNNEncDec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.attention_type = config['attention_type']
        self.alignment_method = config['alignment_method']
        self.context_size = config['context_size']
        self.beam_size = config['beam_size']
        self.strategy = config['decoding_strategy']

        if (self.strategy not in ['topk_sampling', 'beam_search']):
            raise NotImplementedError("{} decoding strategy not implemented".format(self.strategy))

        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.source_token_embedder = nn.Embedding(self.source_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        
        if config['share_vocab']:
            self.target_token_embedder = self.source_token_embedder
        else:
            self.target_token_embedder = nn.Embedding(self.target_vocab_size, self.embedding_size,
                                                      padding_idx=self.padding_token_idx)
        
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
        self.vocab_linear = nn.Linear(self.hidden_size, self.target_vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')
        
        self.max_target_length = config['target_max_seq_length']

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _topk_sampling_generate(self, bid, encoder_states, encoder_outputs, encoder_masks):
        decoder_states = encoder_states[:, bid, :].unsqueeze(1)
        generate_tokens = []
        input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
        for gen_idx in range(self.max_target_length):
            decoder_input = self.target_token_embedder(input_seq)
            if self.attention_type is not None:
                decoder_outputs, decoder_states, _ = self.decoder(decoder_input,
                                                                decoder_states,
                                                                encoder_outputs[bid, :, :].unsqueeze(0),
                                                                encoder_masks[bid, :].unsqueeze(0))
            else:
                decoder_outputs, decoder_states = self.decoder(decoder_input,
                                                            decoder_states)
            token_logits = self.vocab_linear(decoder_outputs)
            token_idx = topk_sampling(token_logits)
            token_idx = token_idx.item()

            if token_idx == self.eos_token_idx:
                break
            else:
                generate_tokens.append(idx2token[token_idx])
                input_seq = torch.LongTensor([[token_idx]]).to(self.device)
        return generate_tokens

    def _beam_search_generate(self, bid, encoder_states, encoder_outputs, encoder_masks):
        decoder_states = encoder_states[:, bid, :].unsqueeze(1)
        encoder_output = encoder_outputs[bid, :, :]
        encoder_mask = encoder_masks[bid, :]
        hypthetic_token_idx = [[self.sos_token_idx]]
        completed_hypotheses = []
        hyp_scores = torch.zeros(1).to(self.device)

        for step in range(self.max_target_length):
            hyp_num = len(hypthetic_token_idx)
            exp_encoder_output = encoder_output.repeat(hyp_num, 1, 1)
            exp_encoder_mask = encoder_mask.repeat(hyp_num, 1)
            input_seq = [hyp[-1] for hyp in hypthetic_token_idx]
            input_seq = torch.tensor(input_seq).to(self.device)
            decoder_input = self.target_token_embedder(input_seq)

            if self.attention_type is not None:
                decoder_outputs, decoder_states, _ = self.decoder(decoder_input, decoder_states, encoder_output, encoder_mask)
            else:
                decoder_outputs, decoder_states = self.decoder(decoder_input, decoder_states)
            token_logits = self.vocab_linear(decoder_outputs)
            token_probs = F.log_softmax(token_logits, dim=-1)
            
            live_hyp_num = self.beam_size - len(completed_hypotheses)
            tmp_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(token_probs) + token_probs).view(-1)
            top_scores, top_pos = torch.topk(tmp_hyp_scores, k=live_hyp_num)
            assert len(self.target_vocab_size) == token_probs.size(-1)
            hyp_ids = top_pos / len(self.target_vocab_size)
            word_ids = top_pos % len(self.target_vocab_size)

            new_hypotheses = []
            new_ids = []
            new_scores = []

            for hyp_id, word_id, score in zip(hyp_ids, word_ids, top_scores):
                new_hyp = hypthetic_token_idx[hyp_id] + [word_id]
                if (word_id == self.eos_token_idx):
                    completed_hypotheses.append((new_hyp[1:-1], score / (step - 1)))
                else:
                    new_hypotheses.append(new_hyp)
                    new_ids.append(hyp_id)
                    new_scores.append(score)
            
            new_ids = torch.tensor(new_ids).to(self.device)
            decoder_states = decoder_states[new_ids]
            hypthetic_token_idx = new_hypotheses
            hyp_scores = torch.tensor(new_scores).to(self.device)
        
        if (len(completed_hypotheses) == 0):
            return hypthetic_token_idx[0][1:]
        else:
            return max(completed_hypotheses, key = lambda hyp: hyp[1])[0]

    def generate(self, eval_dataloader):
        generate_corpus = []
        idx2token = eval_dataloader.target_idx2token
        for batch_data in eval_dataloader:
            source_text = batch_data['source_idx']
            source_length = batch_data['source_length']
            source_embeddings = self.source_token_embedder(source_text)
            encoder_outputs, encoder_states = self.encoder(source_embeddings, source_length)

            if self.bidirectional:
                encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
                encoder_states = encoder_states[0:encoder_states.size(0):2]
                
            encoder_masks = torch.ne(source_text, self.padding_token_idx)
            for bid in range(source_text.size(0)):
                if (self.strategy == 'topk_sampling'):
                    generate_tokens = self._topk_sampling_generate(bid, encoder_states, encoder_outputs, encoder_masks)
                    generate_corpus.append(generate_tokens)
                elif (self.strategy == 'beam_search'):
                    generate_tokens = self._beam_search_generate(bid, encoder_states, encoder_outputs, encoder_masks)
                    generate_corpus.append(generate_tokens)
        
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        source_text = corpus['source_idx']
        source_length = corpus['source_length']

        input_text = corpus['target_idx'][:, :-1]
        target_text = corpus['target_idx'][:, 1:]

        source_embeddings = self.dropout(self.source_token_embedder(source_text))
        input_embeddings = self.dropout(self.target_token_embedder(input_text))
        encoder_outputs, encoder_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
            encoder_states = encoder_states[0:encoder_states.size(0):2]

        encoder_masks = torch.ne(source_text, self.padding_token_idx)

        if self.attention_type is not None:
            decoder_outputs, decoder_states, _ = self.decoder(input_embeddings, encoder_states, encoder_outputs,
                                                              encoder_masks)
        else:
            decoder_outputs, decoder_states = self.decoder(input_embeddings, encoder_states)

        token_logits = self.vocab_linear(decoder_outputs)

        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text.contiguous().view(-1))
        loss = loss.reshape_as(target_text)

        length = corpus['target_length'] - 1
        loss = loss.sum(dim=1) / length
        loss = loss.mean()
        return loss
