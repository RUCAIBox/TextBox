# @Time   : 2021/03/02
# @Author : Yifan Du
# @Email  : yifandu99@outlook.com

r"""
Conditional VAE
################################################
Reference:
    Juntao Li et al. "Generating Classical Chinese Poems via Conditional Variational Autoencoder and Adversarial Training" in ACL 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from textbox.model.abstract_generator import UnconditionalGenerator
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder


class CVAE(UnconditionalGenerator):
    r"""We use the title of a poem and the previous line as condition to generate the current line.
    """

    def __init__(self, config, dataset):
        super(CVAE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.num_highway_layers = config['num_highway_layers']
        self.rnn_type = config['rnn_type']
        self.max_epoch = config['epochs']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.eval_generate_num = config['eval_generate_num']
        self.max_target_length = config['max_target_length']
        self.MLP_neuron_size = config['MLP_neuron_size']  # neuron size in the prior network
        self.max_target_num = config['max_target_num']
        self.gamma = config["gamma"]  # control the lower bound of kl divergence

        self.num_directions = 2 if self.bidirectional else 1
        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        self.encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type, self.dropout_ratio,
            self.bidirectional
        )
        self.decoder = BasicRNNDecoder(
            self.embedding_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        if self.rnn_type == "lstm":
            # prior network
            self.MLP_mean_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_mean_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            self.MLP_logvar_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_logvar_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            # posterior network
            self.hidden_to_mean = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(
                2 * self.num_directions * self.hidden_size + self.latent_size, 2 * self.hidden_size
            )  # first args size=title+pre_line+z
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            self.MLP_mean_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_mean_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            self.MLP_logvar_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_logvar_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            self.hidden_to_mean = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(
                2 * self.num_directions * self.hidden_size + self.latent_size, self.hidden_size
            )
        else:
            raise ValueError("No such rnn type {} for CVAE.".format(self.rnn_type))

        # BN to control the lower bound of kl divergence
        self.posterior_mean_BN = nn.BatchNorm1d(self.latent_size)
        self.posterior_mean_BN.weight.requires_grad = False
        self.posterior_mean_BN.weight.fill_(self.gamma)
        self.posterior_mean_BN.bias.requires_grad = True

        # parameters initialization
        self.apply(self.xavier_uniform_initialization)

    def xavier_uniform_initialization(self, module):
        r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
        nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
        using constant 0 to initialize.
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.uniform_(module.weight.data, a=-0.08, b=0.08)
            # xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight.data, a=-0.08, b=0.08)
            # xavier_uniform_(module.weight.data,gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def generate(self, eval_data):
        generate_corpus = []
        idx2token = eval_data.idx2token
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_data):
                title_text = batch_data['source_text_idx_data']
                title_length = batch_data['source_idx_length_data']
                sentence_length = batch_data['target_idx_length_data']
                sentence_length = torch.Tensor([sentence_length[i][0].item() for i in range(len(sentence_length))])
                batch_size = title_text.size(0)
                pad_text = torch.full((batch_size, self.max_target_length + 2), self.padding_token_idx).to(self.device)
                pad_emb = self.token_embedder(pad_text)
                batch_size = title_text.size(0)
                title_emb = self.token_embedder(title_text)
                title_o, title_hidden = self.encoder(title_emb, title_length)
                pre_o, pre_hidden = self.encoder(pad_emb, sentence_length)
                if self.rnn_type == "lstm":
                    title_h, title_c = title_hidden
                    fir_h, fir_c = pre_hidden
                elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
                    title_h = title_hidden
                    fir_h = pre_hidden
                else:
                    raise NotImplementedError("No such rnn type {} for CVAE.".format(self.rnn_type))
                if self.bidirectional:
                    title_h = title_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                    title_h = title_h[-1]
                    title_h = torch.cat([title_h[0], title_h[1]], dim=1)
                    fir_h = fir_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                    fir_h = fir_h[-1]
                    fir_h = torch.cat([fir_h[0], fir_h[1]], dim=1)
                else:
                    title_h = title_h[-1]
                    fir_h = fir_h[-1]
                for bid in range(batch_size):
                    poem = []
                    pre_h = torch.unsqueeze(fir_h[bid], 0)
                    single_title_h = torch.unsqueeze(title_h[bid], 0)
                    for i in range(self.max_target_num):
                        generate_sentence = []
                        generate_sentence_idx = []
                        condition = torch.cat((single_title_h, pre_h), 1)
                        # mean and logvar of prior：
                        prior_mean = self.MLP_mean_linear1(condition)
                        prior_mean = self.MLP_mean_linear2(F.relu(prior_mean))
                        prior_logvar = self.MLP_logvar_linear1(condition)
                        prior_logvar = self.MLP_logvar_linear2(F.relu(prior_logvar))
                        # sample from prior
                        prior_z = torch.randn([1, self.latent_size]).to(self.device)
                        prior_z = prior_mean + prior_z * torch.exp(0.5 * prior_logvar)
                        hidden = self.latent_to_hidden(torch.cat((condition, prior_z), 1))
                        if self.rnn_type == "lstm":
                            decoder_hidden = torch.chunk(hidden, 2, dim=-1)
                            h_0 = decoder_hidden[0].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                            c_0 = decoder_hidden[1].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                            decoder_hidden = (h_0, c_0)
                        else:
                            decoder_hidden = hidden.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                        input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
                        for _ in range(
                            int(sentence_length[bid].item()) - 2
                        ):  # generate until reach the maximum number of words in a sentence
                            decoder_input = self.token_embedder(input_seq)
                            outputs, hidden_states = self.decoder(
                                input_embeddings=decoder_input, hidden_states=decoder_hidden
                            )
                            token_logits = self.vocab_linear(outputs)
                            token_idx = self.topk_sampling(token_logits)
                            index = 0
                            while token_idx[0][index].item() in [
                                self.sos_token_idx, self.eos_token_idx
                            ]:  # if generate 'eos' before reach the proper number of words, resample
                                index += 1
                            token_idx = token_idx[0][index].item()
                            generate_sentence.append(idx2token[token_idx])
                            generate_sentence_idx.append(token_idx)
                            input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                        poem.append(generate_sentence)
                        generate_sentence_idx = torch.tensor(generate_sentence_idx).to(self.device).to(torch.int64)
                        generate_sentence_length = torch.tensor(len(generate_sentence)).to(self.device).expand(1, 1)
                        pre_emb = self.token_embedder(generate_sentence_idx)
                        pre_emb = torch.unsqueeze(pre_emb, 0)
                        pre_o, pre_hidden = self.encoder(pre_emb, generate_sentence_length[0])
                        if self.rnn_type == "lstm":
                            pre_h, pre_c = pre_hidden
                        else:
                            pre_h = pre_hidden
                        if self.bidirectional:
                            pre_h = pre_h.view(self.num_enc_layers, 2, 1, self.hidden_size)
                            pre_h = pre_h[-1]
                            pre_h = torch.cat([pre_h[0], pre_h[1]], dim=1)
                        else:
                            pre_h = pre_h[-1]
                    generate_corpus.append(poem)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        title_text = corpus['source_text_idx_data']
        title_length = corpus['source_idx_length_data']
        sentence_text = corpus['target_text_idx_data']
        intput_text = sentence_text[:, :, :-1]
        target_text = sentence_text[:, :, 1:]
        sentence_length = corpus['target_idx_length_data']
        sentence_length = torch.Tensor([sentence_length[i][0].item() for i in range(len(sentence_length))])
        batch_size = sentence_text.size(0)

        title_emb = self.token_embedder(title_text)
        sentence_emb = self.token_embedder(sentence_text)
        title_o, title_hidden = self.encoder(title_emb, title_length)
        pad_text = torch.full((batch_size, self.max_target_length + 2),
                              self.padding_token_idx).to(self.device)  # prepare 'pad' to generate the first line
        pad_emb = self.token_embedder(pad_text)
        total_loss = torch.zeros(1).to(self.device)
        total_kld = torch.zeros(1).to(self.device)
        total_rec = torch.zeros(1).to(self.device)
        for i in range(0, self.max_target_num):
            if i == 0:  # there is no previous line for the first line
                pre_o, pre_hidden = self.encoder(pad_emb, sentence_length)
            else:
                pre_o, pre_hidden = self.encoder(sentence_emb[:, i - 1, :, :], sentence_length)
            cur_o, cur_hidden = self.encoder(sentence_emb[:, i, :, :], sentence_length)  # extract the current line
            if self.rnn_type == "lstm":
                title_h, title_c = title_hidden
                pre_h, pre_c = pre_hidden
                cur_h, cur_c = cur_hidden
            elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
                title_h = title_hidden
                pre_h = pre_hidden
                cur_h = cur_hidden
            else:
                raise NotImplementedError("No such rnn type {} for CVAE.".format(self.rnn_type))
            if self.bidirectional:
                title_h = title_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                title_h = title_h[-1]
                title_h = torch.cat([title_h[0], title_h[1]], dim=1)
                pre_h = pre_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                pre_h = pre_h[-1]
                pre_h = torch.cat([pre_h[0], pre_h[1]], dim=1)
                cur_h = cur_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                cur_h = cur_h[-1]
                cur_h = torch.cat([cur_h[0], cur_h[1]], dim=1)
            else:
                title_h = title_h[-1]
                pre_h = pre_h[-1]
                cur_h = cur_h[-1]
            # concatenate the title and the previous line
            condition = torch.cat((title_h, pre_h), 1)
            # prior network
            prior_mean = self.MLP_mean_linear1(condition)
            prior_mean = self.MLP_mean_linear2(F.relu(prior_mean))
            prior_logvar = self.MLP_logvar_linear1(condition)
            prior_logvar = self.MLP_logvar_linear2(F.relu(prior_logvar))
            # posterior network
            posterior_mean = self.hidden_to_mean(cur_h)
            posterior_logvar = self.hidden_to_logvar(cur_h)
            posterior_mean = self.posterior_mean_BN(posterior_mean)
            # sample from the posterior
            posterior_z = torch.randn([batch_size, self.latent_size]).to(self.device)
            posterior_z = posterior_mean + posterior_z * torch.exp(0.5 * posterior_logvar)
            hidden = self.latent_to_hidden(torch.cat((condition, posterior_z), 1))

            if self.rnn_type == "lstm":
                decoder_hidden = torch.chunk(hidden, 2, dim=-1)
                h_0 = decoder_hidden[0].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                c_0 = decoder_hidden[1].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                decoder_hidden = (h_0, c_0)
            else:
                decoder_hidden = hidden.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
            input_emb = sentence_emb[:, i, :-1, :]
            input_emb = self.dropout(input_emb)  # add dropout to weaken the decoder
            outputs, hidden_states = self.decoder(input_embeddings=input_emb, hidden_states=decoder_hidden)
            token_logits = self.vocab_linear(outputs)
            loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text[:, i, :].contiguous().view(-1))
            loss = loss.reshape_as(target_text[:, i, :])
            length = (torch.as_tensor(sentence_length, dtype=torch.float32) - 1).to(self.device)
            loss = loss.sum(dim=1) / length
            kld = 0.5 * torch.sum(
                prior_logvar - posterior_logvar - 1 + torch.exp(posterior_logvar) / torch.exp(prior_logvar) +
                (prior_mean - posterior_mean).pow(2) / torch.exp(prior_logvar), 1
            )
            # cycling weight
            kld_coef = float(epoch_idx % 8) / self.max_epoch
            if epoch_idx > self.max_epoch - 8:
                kld_coef = float(epoch_idx / self.max_epoch)
            loss = loss.mean() + kld_coef * kld.mean()
            total_loss += loss

        return total_loss

    def calculate_nll_test(self, corpus, epoch_idx=0):
        title_text = corpus['source_text_idx_data']
        title_length = corpus['source_idx_length_data']
        sentence_text = corpus['target_text_idx_data']
        intput_text = sentence_text[:, :, :-1]
        target_text = sentence_text[:, :, 1:]
        sentence_length = corpus['target_idx_length_data']
        sentence_length = torch.Tensor([sentence_length[i][0].item() for i in range(len(sentence_length))])
        batch_size = sentence_text.size(0)

        title_emb = self.token_embedder(title_text)
        sentence_emb = self.token_embedder(sentence_text)
        title_o, title_hidden = self.encoder(title_emb, title_length)
        total_loss = torch.zeros(1).to(self.device)
        for i in range(1, self.max_target_num):
            pre_o, pre_hidden = self.encoder(sentence_emb[:, i - 1, :, :], sentence_length)
            cur_o, cur_hidden = self.encoder(sentence_emb[:, i, :, :], sentence_length)
            if self.rnn_type == "lstm":
                title_h, title_c = title_hidden
                pre_h, pre_c = pre_hidden
                cur_h, cur_c = cur_hidden
            elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
                title_h = title_hidden
                pre_h = pre_hidden
                cur_h = cur_hidden
            else:
                raise NotImplementedError("No such rnn type {} for CVAE.".format(self.rnn_type))
            if self.bidirectional:
                title_h = title_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                title_h = title_h[-1]
                title_h = torch.cat([title_h[0], title_h[1]], dim=1)
                pre_h = pre_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                pre_h = pre_h[-1]
                pre_h = torch.cat([pre_h[0], pre_h[1]], dim=1)
                cur_h = cur_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                cur_h = cur_h[-1]
                cur_h = torch.cat([cur_h[0], cur_h[1]], dim=1)
            else:
                title_h = title_h[-1]
                pre_h = pre_h[-1]
                cur_h = cur_h[-1]
            # concatenate title and the previous line
            condition = torch.cat((title_h, pre_h), 1)
            prior_mean = self.MLP_mean_linear1(condition)
            prior_mean = self.MLP_mean_linear2(F.relu(prior_mean))
            prior_logvar = self.MLP_logvar_linear1(condition)
            prior_logvar = self.MLP_logvar_linear2(F.relu(prior_logvar))
            posterior_mean = self.hidden_to_mean(cur_h)
            posterior_logvar = self.hidden_to_logvar(cur_h)

            posterior_z = torch.randn([batch_size, self.latent_size]).to(self.device)
            posterior_z = posterior_mean + posterior_z * torch.exp(0.5 * posterior_logvar)
            hidden = self.latent_to_hidden(torch.cat((condition, posterior_z), 1))

            if self.rnn_type == "lstm":
                decoder_hidden = torch.chunk(hidden, 2, dim=-1)
                h_0 = decoder_hidden[0].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                c_0 = decoder_hidden[1].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                decoder_hidden = (h_0, c_0)
            else:
                decoder_hidden = hidden.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
            input_emb = sentence_emb[:, i, :-1, :]
            outputs, hidden_states = self.decoder(input_embeddings=input_emb, hidden_states=decoder_hidden)
            token_logits = self.vocab_linear(outputs)
            loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text[:, i, :].contiguous().view(-1))
            loss = loss.reshape_as(target_text[:, i, :])
            loss = loss.sum(dim=1)
            total_loss += loss.mean()
        return total_loss

    def topk_sampling(self, logits, temperature=1.0, top_k=0, top_p=0.9):
        r"""
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

        Args:
            logits (torch.Tensor): logits distribution
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

        Return:
            torch.Tensor: the top3 chosen index of token for word generating.
        """
        logits = logits / temperature
        top_k = min(top_k, logits.size(-1))  # Safety check

        if top_k > 0:
            values = torch.topk(logits, top_k)[0]  # B x top_k
            batch_mins = values[:, :, -1].expand_as(logits.squeeze(1)).unsqueeze(1)
            logits = torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

        if 0.0 < top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

            cumprobs = sorted_probs.cumsum(dim=-1)

            # Create mask for all cumulative probabilities less than p
            mask = cumprobs < top_p

            # First mask must always be pickable
            mask = F.pad(mask[:, :, :-1], (1, 0, 0, 0), value=1)

            masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')).to(probs))

            batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)

            # Mask out all logits (tail) that are too small
            logits = torch.where(probs < batch_mins, torch.tensor(float('-inf')).to(logits), logits)

        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities.squeeze(1)
        token_idx = torch.multinomial(probabilities, 3)

        return token_idx
