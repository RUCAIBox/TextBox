# @Time   : 2021/12/14
# @Author : Junjie Zhang
# @Email  : jjzhang_233@stu.xidian.edu.cn

r"""
data2textencdec
################################################
Reference:
    Puduppully et al. " Data-to-Text Generation with Content Selection and Planning" in AAAI 2019.
"""
from __future__ import division
import os
from collections import Counter
import torch
import torch.nn as nn
import torchtext
from textbox.module.strategy import CopyGeneratorCriterion
from textbox.model.abstract_generator import TwoStageGenerator
from textbox.module.ModelConstructor import make_base_model，TranslationBuilder, Translator, GNMTGlobalScorer
from textbox.model.init import  xavier_normal_initialization
TGT_VOCAB_SIZE = 606

class data2textencdec(TwoStageGenerator):
    def __init__(self,config,dataset):
        super(data2textencdec, self).__init__(config, dataset)

        # load parameters info
        self.config = config
        self.model1,self.model2 = self.build_model(dataset,config)
        self.use_gpu = config['use_gpu']
        self.target_idx2token = dataset.target_idx2token
        self.target_token2idx = dataset.target_token2idx
        self.target_vocab_size = dataset.target_vocab_size
        self.source_idx2token = dataset.source_idx2token
        self.source_token2idx = dataset.source_token2idx
        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx
        self.beam_size = config['beam_size']
        self.batch_size = config['train_batch_size']
        # define loss
        weight = torch.ones(TGT_VOCAB_SIZE)
        weight[self.padding_token_idx] = 0
        self.criterion1 = nn.NLLLoss(weight, reduction='sum')
        self.switch_loss_criterion = nn.BCELoss(reduction='sum')
        self.criterion2 = CopyGeneratorCriterion(self.target_vocab_size,
                                                self.padding_token_idx)
        # parameters initialization
        self.apply(xavier_normal_initialization)
        
    # define layers
    def build_model(self,dataset,config):
        model1 = make_base_model(config, dataset.source_idx2token,dataset.source_token2idx,dataset.target_idx2token,
                              stage1=True)
        model2 = make_base_model(config,dataset.source_idx2token,dataset.source_token2idx,dataset.target_idx2token,
                              stage1=False)
        return model1, model2

    def generate(self, batch_data, eval_data):
        generate_corpus = []
        scorer = GNMTGlobalScorer(0.,-0.,'none', 'none')
        stage1 = True
        self.batch_size = batch_data['target_idx'].size(0)
        translator = Translator(
            self.model1, self.model2, self.sos_token_idx,self.padding_token_idx,self.eos_token_idx,self.target_vocab_size,
            batch_size = self.batch_size,
            beam_size=self.beam_size,
            n_best=1,
            global_scorer=scorer,
            max_length=80,
            copy_attn= False,
            cuda=self.use_gpu,
            beam_trace=False,
            min_length=35
            )

        builder = TranslationBuilder(
            eval_data, self.eos_token_idx,batch_size = self.batch_size,
            has_tgt=False)
        trans_batch = translator.translate_batch(batch_data,eval_data,stage1)
        translations = builder.from_batch(trans_batch,batch_data,stage1)
        content_plans = []
        source_plan_length = []
        for trans in translations:
            for pred in trans.pred_sents[:1]:
                source_plan_length.append(len(pred))
                pred = torch.tensor(pred)
                content_plans.append(pred)
        source_plan_length = torch.tensor((source_plan_length)).type_as(batch_data["target_length"]).to(self.device)
        source_plan_idx = torch.zeros(self.batch_size,source_plan_length.max(),dtype=torch.int64)

        for i,source_plan in enumerate(content_plans):
            source_plan_idx[i,:source_plan_length[i]] = content_plans[i]
        content_plans = source_plan_idx
        content_plans_src_map = content_plans
        # add eos,bos to source_plan_idx
        content_plan = self.add_eos_bos(content_plans,source_plan_length)
        batch_data['source_plan_idx'] = content_plan
        batch_data['source_plan_length'] = source_plan_length + 2
        # make src2map
        content_plans = content_plans_src_map
        content_lengths = batch_data['source_plan_length'] - 2
        source_text = batch_data['source_text']
        alignment, src_vocabs = self.get_src_map(content_plans,content_lengths,source_text)
        batch_data['src_map'] = alignment.to(self.device)
        batch_data['src_vocabs'] = src_vocabs
        batch_data['alignment'] = self.get_alignment(batch_data['target_text'],src_vocabs).to(self.device)
        # stage2
        translator.copy_attn = True
        stage1 = False
        translator.max_length = 850
        translator.min_length = 150
        trans_batch = translator.translate_batch(batch_data, eval_data, stage1)
        translations = builder.from_batch(trans_batch, batch_data,stage1)
        for trans in translations:
            generate_tokens = trans.pred_sents[0]
            generate_corpus.append(generate_tokens)
        return generate_corpus

    def forward(self, corpus, epoch_idx=0):
        self.batch_size = corpus['target_idx'].size(0)
        source_text = corpus['source_text']
        content_plans = corpus['source_plan_idx']
        content_plans_src_map = corpus['source_plan_idx']
        # add eos,bos to source_plan_idx
        content_plan = self.add_eos_bos(content_plans,corpus['source_plan_length'])
        corpus['source_plan_idx'] = content_plan
        corpus['source_plan_length'] = corpus['source_plan_length']+2
        content_plans = content_plans_src_map
        content_lengths = corpus['source_plan_length'] - 2
        alignment, src_vocabs = self.get_src_map(content_plans,content_lengths,source_text)
        corpus['src_map'] = alignment.to(self.device)
        corpus['src_vocabs'] = src_vocabs

        # stage1
        dec_state = None
        src = corpus['source_idx'].transpose(0,1).to(self.device)
        src_lengths = torch.LongTensor(src.size(1)).fill_(src.size(0)).to(self.device)
        tgt = corpus['source_plan_idx'].transpose(0,1).contiguous().unsqueeze(2).to(self.device)
        self.model1.zero_grad()
        outputs, attns, dec_state, memory_bank = \
            self.model1(src, tgt, src_lengths, dec_state)
        scores = self._bottle(attns['std'])
        target = corpus['source_plan_idx'].transpose(0,1).contiguous()
        target = target[1:]
        gtruth = target.view(-1)
        loss1 = self.criterion1(scores, gtruth)
        loss1 = loss1 / self.batch_size
        loss1 = loss1 / target.size(1)
        if dec_state is not None:
            dec_state.detach()

        # stage2
        dec_state = None
        inp_stage2 = tgt[1:-1]
        index_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                        zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(inp_stage2, 2)))]
        emb = torch.transpose(torch.cat(index_select), 0, 1)
        tgt_outer = corpus['target_idx'].transpose(0,1).contiguous().unsqueeze(2)
        tgt = tgt_outer
        # alignment
        corpus['alignment'] = self.get_alignment(corpus['target_text'], src_vocabs).to(self.device)
        src_lengths = corpus['source_plan_length']
        src_lengths = src_lengths-2
        self.model2.zero_grad()
        outputs, attns, dec_state, _ = \
            self.model2(emb, tgt, src_lengths, dec_state)
        copy_attn = attns.get("copy")
        align = corpus['alignment'][1:]
        align = align.view(-1)
        target = corpus['target_idx'].transpose(0,1).contiguous()[1:]
        target = target.view(-1)

        scores, p_copy = self.model2.generator(self._bottle(outputs),
                                              self._bottle(copy_attn),
                                              corpus['src_map'],align)
        loss = self.criterion2(scores,align,target)
        switch_loss = self.switch_loss_criterion(p_copy,align.ne(0).float().view(-1,1))
        switch_loss = switch_loss / int(align.size(0))
        loss = loss.sum()
        loss = loss / int(align.size(0))
        loss2 = loss + switch_loss
        return loss1,loss2

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))

    def add_eos_bos(self,content_plans,source_plan_length):
        src_len = content_plans.size(1)
        content_plan = torch.zeros(content_plans.size(0), src_len + 2).type_as(content_plans)
        content_plan[:, 0] = 2
        for i in range(content_plans.size(0)):
            j = source_plan_length[i]
            if type(j) != 'int':
                j = j.item()
            content_plan[i, 1:j + 1] = content_plans[i, 0:j]
            content_plan[i, j + 1] = 3
            content_plan[i, j + 2:] = 0
        content_plan = content_plan.type_as(content_plans)
        return content_plan

    def get_src_map(self, content_plans, content_lengths, source_text):
        src2 = []
        for i, source in enumerate(source_text):
            content_plan = content_plans[i]
            output = []
            for j, record in enumerate(content_plan):
                if j >= content_lengths[i]:
                    break
                elements = source[int(record)][0]
                output.append(elements)
            src2.append(output)
        src_vocabs = []
        data = []
        for i, example in enumerate(src2):
            src = tuple(example)
            src_vocab = torchtext.vocab.Vocab(Counter(src), specials=["<|pad|>", "<|unk|>"])
            src_vocabs.append(src_vocab)
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            data.append(src_map)
        src_size = max([t.size(0) for t in data])
        src_vocab_size = max([t.max() for t in data]) + 1
        alignment = torch.zeros(src_size, len(data), src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                alignment[j, i, t] = 1
        return alignment, src_vocabs

    def get_alignment(self, target_text, src_vocabs):
        masks = []
        for i, tgt2 in enumerate(target_text):
            src_vocab = src_vocabs[i]
            mask = torch.LongTensor(
                [0] + [src_vocab.stoi[w] for w in tgt2] + [0])
            masks.append(mask)
        data = masks
        tgt_size = max([t.size(0) for t in data])
        alignment = torch.zeros(tgt_size, len(data)).long()
        for i, sent in enumerate(data):
            alignment[:sent.size(0), i] = sent
        return alignment










