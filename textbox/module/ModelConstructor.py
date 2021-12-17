# @Time   : 2021/12/14
# @Author : Junjie Zhang
# @Email  : jjzhang_233@stu.xidian.edu.cn


"""
textbox.module.ModelConstructor
make model for table2text
Contains Decoder, Encoder, Embeddings, BeamSearch, TextGeneration module 
"""
from __future__ import division, unicode_literals
import torch
import torch.nn as nn
from torch.autograd import Variable
from textbox.module.strategy import CopyGenerator
from textbox.utils.enum_type import SpecialTokens
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from textbox.module.Attention.PointerAttention import GlobalSelfAttention,GlobalAttention


def make_embeddings(config, vocab_size, feat_vocab_size,word_padding_idx):
    """
    Make an Embeddings instance.
    """
    embedding_dim = config['embedding_size']
    num_word_embeddings = vocab_size
    feats_padding_idx = [word_padding_idx for i in range(len(feat_vocab_size))]
    num_feat_embeddings = feat_vocab_size
    return Embeddings(embedding_dim=embedding_dim,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings)


def make_encoder(config, embeddings=None, stage1=True):
    """
    Various encoder dispatcher function.
    """

    if stage1:
        return MeanEncoder(config['num_enc_layers1'], embeddings, config['embedding_size'], 0, config['dropout_ratio'])
    else:
        return RNNEncoder(config['rnn_type'], config['bidirectional'], config['num_enc_layers2'],
                          config['hidden_size'], config['dropout_ratio'], embeddings)


def make_decoder(config, embeddings, stage1):
    """
    Various decoder dispatcher function.
    """
    if stage1:
        return PointerRNNDecoder(config['rnn_type'], config['bidirectional'], config['num_dec_layers1'],
                                 config['hidden_size'],'general',False, None, False,
                                 config['dropout_ratio'], embeddings, False, 'pointer')
    else:
        return InputFeedRNNDecoder(config['rnn_type'], config['bidirectional'], config['num_dec_layers2'],
                                   config['hidden_size'], 'general', False, None, True, config['dropout_ratio'],
                                   embeddings, True)


def make_base_model(config, source_idx2token,source_token2idx,target_idx2token,  stage1=True):
    """
    return the final model
    """
    gpu = config['use_gpu']
    if stage1:
        src_dict = source_idx2token[0]
        src_vocab_size = len(src_dict)
        word_padding_idx = 0
        src_feat_vocab_size = [len(source_idx2token[i]) for i in range (1,2)]
        src_embeddings = make_embeddings(config, src_vocab_size,src_feat_vocab_size,
                                        word_padding_idx )
        encoder = make_encoder(config, src_embeddings, stage1)
    else:
        encoder = make_encoder(config,None,stage1)
    # Make decoder.
    if not stage1:
        tgt_dict = target_idx2token
        tgt_vocab_size = len(tgt_dict)
        word_padding_idx = 0
        tgt_feat_vocab_size = []
        tgt_embeddings = make_embeddings(config, tgt_vocab_size, tgt_feat_vocab_size,
                                         word_padding_idx)
        decoder = make_decoder(config, tgt_embeddings, stage1)
    else:
        decoder = make_decoder(config, None, stage1)
    model = NMTModel(encoder, decoder)
    model.model_type = 'text'
    # Make Generator.
    if stage1:
        generator = None
    else:
        generator = CopyGenerator(config['hidden_size'],
                                  target_idx2token)
    print('Intializing model parameters.')
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)
    if not stage1:
        for p in generator.parameters():
            p.data.uniform_(-0.1, 0.1)
    model.generator = generator
    return model


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.

    Args:
        embedding_dim (int): embedding size
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
    """
    def __init__(self, embedding_dim, word_vocab_size, word_padding_idx,
                 feat_padding_idx=[], feat_vocab_sizes=[]):

        self.word_padding_idx = word_padding_idx
        vocab_sizes = [word_vocab_size]
        emb_dims = [embedding_dim]
        pad_indices = [word_padding_idx]
        feat_dims = [embedding_dim] * len(feat_vocab_sizes)
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(int(vocab), int(dim), padding_idx=pad)
                      for vocab, dim, pad in emb_params]
        feat_merge = "mlp"  # mlp
        emb_luts = Elementwise(feat_merge, embeddings)
        self.embedding_size = embedding_dim
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)
        if feat_merge == 'mlp' and len(feat_vocab_sizes)>0:
            in_dim = sum(emb_dims)
            out_dim = embedding_dim
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

    def forward(self, input):
        emb = self.make_embedding(input)
        return emb

class EncoderBase(nn.Module):
    r"""
    Base encoder class. Specifies the interface used by different encoder types
    """

    def forward(self, src, lengths=None, encoder_state=None):
        r"""
        Args:
            src (Torch.LongTensor): padded sequences of sparse indices, shape:[src_len x batch x n_feat]
            lengths (Torch.LongTensor`): length of each sequence, shape:[batch]
            encoder_state (rnn-class specific): initial encoder_state state.

        Returns:
            Torch.FloatTensor: final encoder state, used to initialize decoder
            Torch.FloatTensor: memory bank for attention, shape:`[src_len x batch x hidden]
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    r"""A trivial non-recurrent encoder. Simply applies mean pooling,which will be user in stage1.
    Args:
       num_layers (int): number of replicated layers
       embeddings : embedding module to use
    """
    def  __init__(self, num_layers, embeddings, emb_size, attn_hidden, dropout=0.0, attn_type="general",
                 coverage_attn=False):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=dropout)
        self.attn = GlobalSelfAttention(emb_size, coverage=coverage_attn, attn_type=attn_type, attn_hidden=attn_hidden)

    def forward(self, src, lengths=None, encoder_state=None, memory_lengths=None):
        r"""
        Args and returns like EncoderBase
        """
        src = src[:, :, :-2] # get the first and second feature
        emb = self.dropout(self.embeddings(src))
        nan = torch.isnan(emb)
        isnan  = sum(sum(sum(nan)))
        if isnan or torch.isnan(emb[0][0][0]):
            emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        decoder_output, p_attn = self.attn(emb.transpose(0, 1).contiguous(), emb.transpose(0, 1), memory_lengths=lengths)
        mean = decoder_output.mean(0).expand(self.num_layers, batch, emb_dim).contiguous() # self.num_layers: 1
        memory_bank = decoder_output
        encoder_final = (mean, mean)
        return encoder_final, memory_bank


class RNNEncoder(EncoderBase):
    r""" A generic recurrent neural network encoder,which will be user in stage2.
    Args:
       rnn_type (str): style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings : embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        num_directions = 2  if bidirectional else 1
        output_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False
        self.rnn = getattr(nn, rnn_type)(input_size=hidden_size, hidden_size=output_size,
                                         num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, src, lengths=None, encoder_state=None):
        r"""
        Args and Returns like EncoderBase
                """
        emb = src
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths,enforce_sorted=False)
        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        return encoder_final, memory_bank


class RNNDecoderBase(nn.Module):
    r"""
    Base recurrent attention-based decoder class.
    Args:
       rnn_type (:obj:`str`): style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :'textbox.modules.PointerAttention`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings : embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers, hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None, copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, pointer_decoder_type = None):
        super(RNNDecoderBase, self).__init__()
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        if pointer_decoder_type == 'pointer':
            self.bidirectional_encoder = False
        self.rnn = self._build_rnn(rnn_type, input_size=self._input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout)
        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = GlobalAttention("GlobalAttention", hidden_size, coverage=coverage_attn, attn_type=attn_type)
        if pointer_decoder_type == 'pointer':
            self.attn = GlobalAttention("PointerAttention", hidden_size)
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = GlobalAttention("GlobalAttention",
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (Torch.LongTensor): sequences of padded tokens,shape:[tgt_len x batch x n_feats]
            memory_bank (Torch.FloatTensor`): vectors from the encoder,shape:[src_len x batch x hidden]. （r_j^cs）
            state :decoder state object to initialize the decoder
            memory_lengths (Torch.LongTensor`): the padded source lengths, shape:[batch].
        Returns:
            Torch.FloatTensor: decoder_outputs: output from the decoder (after attn),shape:[tgt_len x batch x hidden]`.
            decoder_state: final hidden state from the decoder
            Torch.FloatTensor: attns: distribution over src at each tgt, shape:[tgt_len x batch x src_len]
        """
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)
        if decoder_outputs is None:
            final_output = None
        else:
            final_output = decoder_outputs[-1].unsqueeze(0)
        coverage = None
        state.update_state(decoder_final, final_output, coverage)
        if decoder_outputs is not None:
            # Concatenates sequence of tensors along a new dimension.
            decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            if type(attns[k]) == list:
                attns[k] = torch.stack(attns[k])
        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

class PointerRNNDecoder(RNNDecoderBase):
    r"""
    See : RNNDecoderBase for options. which will be used in stage2
    Based around the approach from "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        r"""
        Args:
            tgt (Torch.LongTensor): a sequence of input tokens tensors,shape:[len x batch x nfeats].
            memory_bank (Torch.FloatTensor): output(tensor sequence) from the encoder
                        RNN of size,shape: [src_len x batch x hidden_size]
            state (Torch.FloatTensor): hidden state from the encoder RNN for initializing the decoder.
            memory_lengths (Torch.LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final: final hidden state from the decoder.
            Torch.FlaotTenosr: decoder_outputs, an array of output of every time step from the decoder.
            dict of (str, [Torch.FloatTensor]: attns ,a dictionary of different
                            type of attention Tensor array of every time step from the decoder.
        """
        # Initialize local and return variables.
        attns = {}
        emb = torch.transpose(torch.cat(
            [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
             zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(tgt,2)))]), 0, 1)
        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb.contiguous(), state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb.contiguous(), state.hidden)
        # Calculate the attention.

        p_attn = self.attn(rnn_output.transpose(0, 1).contiguous(),
                           memory_bank.transpose(0, 1), memory_lengths=memory_lengths )
        attns["std"] = p_attn
        return decoder_final, None, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn = getattr(nn,rnn_type)(**kwargs)
        return rnn

    @property
    def _input_size(self):
        r"""
        Private helper returning the number of expected features.
        """
        return self.hidden_size


class InputFeedRNNDecoder(RNNDecoderBase):
    r"""
    Input feeding based decoder,which will be used in stage1. See :RNNDecoderBase for options.
    Based around the input feeding approach from "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        r"""
        See : PointerRNNEncoder for Args and Returns.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
        emb = self.embeddings(tgt)
        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, hidden = self.rnn(decoder_input, hidden)


            decoder_output, p_attn = self.attn(rnn_output,
                                               memory_bank.transpose(0, 1), memory_lengths=memory_lengths)
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output
            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]
            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn('GlobalAttention', decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        else:
            stacked_cell = StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size

class NMTModel(nn.Module):
    r"""
     Implements a trainable interface for a simple, generic encoder + decoder model.
    Args:
      encoder : an encoder object
      decoder : a decoder object
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        r"""Forward propagate a `src` and `tgt` pair for training.
        Args:
            src (Torch.LongTensor`):  a source sequence passed to encoder.shape:[len x batch x features]`.
            tgt (Torch.LongTensor`): a target sequence of size `[tgt_len x batch]`.
            lengths(torch.LongTensor): the src lengths, pre-padding shape:`[batch]`.
            dec_state (DecoderState, optional): initial decoder state
        Returns:
            Torch.FloatTensor: decoder output, shape: [tgt_len x batch x hidden]
            dict: dictionary attention dists of `[tgt_len x batch x src_len]`
            decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state, memory_bank

class DecoderState(object):
    r"""
    Interface for grouping together the current state of a recurrent decoder.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        r"""
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple): # 用最后一个时间步的hidden_state,更新 self.hidden
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        if input_feed is not None:
            self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        with torch.no_grad():
            vars = [Variable(e.data.repeat(1, beam_size, 1)) for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

class StackedLSTM(nn.Module):
    """
    an implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding, which didn't match standard lstm.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    """
        an implementation of stacked LSTM.
        Needed for the decoder, because we do input feeding, which didn't match standard GRU.
        """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, (h_1,)

class TranslationBuilder(object):
    r"""
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.
    """
    def __init__(self, data, eos_token_idx, batch_size,n_best=1, replace_unk=False,
                 has_tgt=False):
        self.data = data
        self.eos_token_idx = eos_token_idx
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt
        self.batch_size = batch_size

    def _build_target_tokens(self,  src_vocab,  pred, attn, stage1,tgt2_vocab_size,idx2token):
        tokens = []
        for tok in pred:
            if stage1:
                tokens.append(tok.item())
            else:
                tok = tok.item()
                if tok < tgt2_vocab_size:
                    str = idx2token[tok]
                    l = str.split('_')
                    tokens.extend(l)
                else:
                    tokens.append(src_vocab.itos[tok-tgt2_vocab_size])
            if (stage1 and tokens[-1] == self.eos_token_idx) or (not stage1 and tokens[-1] == SpecialTokens.EOS):
                tokens = tokens[:-1]
                break
        return tokens

    def from_batch(self, translation_batch, batch_data,stage1=False):
        batch_size = self.batch_size
        preds = translation_batch["predictions"]
        pred_score = translation_batch["scores"]
        attn = translation_batch["attention"]
        gold_score =  translation_batch["gold_score"]
        data_type = 'text'
        src = None
        translations = []
        for b in range(batch_size):
            if data_type == 'text' and not stage1:
                src_vocab = batch_data['src_vocabs'][b]
            else:
                src_vocab = None
            pred_sents = [self._build_target_tokens(
                src_vocab,
                preds[b][n], attn[b][n], stage1,self.data.target_vocab_size, self.data.target_idx2token)
                          for n in range(self.n_best)]
            gold_sent = None
            translation = Translation(src[:, b] if src is not None else None,
                                     pred_sents, attn[b], pred_score[b], gold_sent, gold_score[b])
            translations.append(translation)
        return translations


class Translation(object):
    r"""
    Container for a translated sentence.
    Attributes:
        src (Torch.LongTensor): src word ids
        src_raw ([str]): raw src words
        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns (list[Torch.FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """
    def __init__(self, src,  pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
    def log(self, sent_number):
        """
        Log translation to stdout.
        """
        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)
        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        print("PRED SCORE: {:.4f}".format(best_score))
        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}".format(self.gold_score))
        if len(self.pred_sents) > 1:
            print('\nBEST HYP:')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)
        return output


class Translator(object):
    r"""
    Uses a model to translate a batch of sentences.
    """
    def __init__(self, model, model2, sos,pad,eos,tgt2_vocab_size,batch_size,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 cuda=False,
                 beam_trace=False,
                 min_length=0,
                 stepwise_penalty=False):
        self.model = model
        self.model2 = model2
        self.sos = sos
        self.pad = pad
        self.eos = eos
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.batch_size = batch_size
        self.tgt2_vocab_size = tgt2_vocab_size
        # for debugging
        self.beam_accum = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}
    def translate_batch(self, batch, data, stage1):
        """
        Translate a batch of sentences.
        """
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = self.batch_size
        data_type = 'text'
        token2idx = data.target_token2idx
        beam = [Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=self.pad,
                                    eos=self.eos,
                                    bos=self.sos,
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty)
                for __ in range(batch_size)]
        # Help functions for working with beams and batches

        def var(a):
            with torch.no_grad():
                x = Variable(a)
                return x

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = batch['source_idx'].transpose(0,1).contiguous()
        src_lengths = batch['source_length'].to(self.device)
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        # memory_bank is r_j^cs: [ src1_len x batch_size x embedding_size]
        src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data) \
            .long() \
            .fill_(memory_bank.size(0))
        model = self.model
        if not stage1:
            if data_type == 'text':
                src_lengths = batch['source_plan_length']-2
            inp_stage2 = batch['source_plan_idx'].transpose(0,1).contiguous().unsqueeze(2)[1:-1]
            index_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                            zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(inp_stage2, 2)))]
            emb = torch.transpose(torch.cat(index_select), 0, 1)
            enc_states, memory_bank = self.model2.encoder(emb, src_lengths)
            model = self.model2
        dec_states = model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)
        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch['src_map'].data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        # memory_lengths : [batch_size * beam_size]
        dec_states.repeat_beam_size_times(beam_size)
        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break
            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1)).to(self.device)
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(self.tgt2_vocab_size - 1), 0)
            inp = inp.unsqueeze(2)
            # Run one step.
            dec_out, dec_states, attn = model.decoder(
                inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            if not stage1:
                dec_out = dec_out.squeeze(0)
            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                if stage1:
                    upd_attn = unbottle(attn["std"]).data
                    out = upd_attn
                    # [beam , batch , src_len]
                else:
                    out = model.generator.forward(dec_out).data
                    out = unbottle(out)
                    # beam x tgt_vocab
                    beam_attn = unbottle(attn["std"])
            else:

                out = model.generator.forward(dec_out,attn["copy"].squeeze(0),
                                              src_map)
                # out[0]: prob [ beam*batch , tgt2_vocab_size + src2_vocab_size]
                out = self.collapse_copy_scores(
                    unbottle(out[0].data), # [beam,batch,tgt2_vocab_size+ src2_vocab_size]
                    token2idx, batch['src_vocabs'])

                out = out.log()   # beam x batch x tgt2_vocab_size+src2_vocab_size
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                if stage1:
                    b.advance(
                        out[:, j],
                        torch.exp(unbottle(attn["std"]).data[:, j, :memory_lengths[j]]))
                else:

                    b.advance(out[:, j],  # [ beam, tgt2_vocab+size+src2_vocab_size]
                        beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)
        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)

            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def collapse_copy_scores(self, scores,  token2idx, src_vocabs):
        r"""
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = self.tgt2_vocab_size
        for b in range(self.batch_size):
            blank = []
            # blank_src = []
            fill = []
            src_vocab = src_vocabs[b]
            for i in range(0, len(src_vocab)):
                if i == 1:  # remove unk
                    continue
                sw = src_vocab.itos[i]
                if token2idx.get(sw):
                    ti = token2idx[sw]
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).long().to(self.device)
                fill = torch.Tensor(fill).long().to(self.device)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
            scores[:,b,offset:] = 1e-10
        return scores

class Beam(object):
    """
    Class for managing the internals of the beam search process.
    """
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0, stepwise_penalty=False):

        self.size = size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scores = torch.FloatTensor(size).zero_().to(self.device)
        self.all_scores = []
        # The backpointers at each time-step.
        self.prev_ks = []
        # The outputs at each time-step.
        self.next_ys = [torch.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False
        # The attentions (matrix) for each time.
        self.attn = []
        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best
        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}
        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1) # tgt2_vocab_size + src2_vocab_size
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1) # tgt2_vocab_size + src2_vocab_size
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = torch.div( best_scores_id , num_words, rounding_mode='floor')
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)
        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        r"""
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """
    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(cov_penalty,
                                                   length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attentions"
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty


class PenaltyBuilder(object):
    """
    Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen
    """
    def __init__(self, cov_pen, length_pen):
        self.length_pen = length_pen
        self.cov_pen = cov_pen

    def coverage_penalty(self):
        if self.cov_pen == "wu":
            return self.coverage_wu
        elif self.cov_pen == "summary":
            return self.coverage_summary
        else:
            return self.coverage_none

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    """
    Below are all the different penalty terms implemented so far
    """

    def coverage_wu(self, beam, cov, beta=0.):
        """
        NMT coverage re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """
        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        return beta * penalty

    def coverage_summary(self, beam, cov, beta=0.):
        """
        Our summary penalty.
        """
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(1)
        penalty -= cov.size(1)
        return beta * penalty

    def coverage_none(self, beam, cov, beta=0.):
        """
        returns zero as penalty
        """
        return beam.scores.clone().fill_(0.0)

    def length_wu(self, beam, logprobs, alpha=0.):
        """
        NMT length re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        modifier = (((5 + len(beam.next_ys)) ** alpha) /
                    ((5 + 1) ** alpha))
        return (logprobs / modifier)

    def length_average(self, beam, logprobs, alpha=0.):
        """
        Returns the average probability of tokens in a sequence.
        """
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0., beta=0.):
        """
        Returns unmodified scores.
        """
        return logprobs
