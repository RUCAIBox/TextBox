import logging
import json
import spacy
from collections import OrderedDict
from . import utils
from . import ontology
from .db_ops import MultiWozDB


class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None
        self.set_stats = {}

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5:
                del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)


class MultiWozReader(_ReaderBase):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.db = MultiWozDB(self.cfg.dbs)
        self.nlp = spacy.load('en_core_web_sm')
        self.vocab_size = self._build_vocab()

        self.domain_files = json.loads(open(self.cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(open(self.cfg.slot_value_set_path, 'r').read())

        self.exp_files = {}
        self._load_data()

    def _build_vocab(self):
        self.vocab = utils.Vocab(self.cfg.vocab_size)
        vp = self.cfg.vocab_path_train
        self.vocab.load_vocab(vp)
        return self.vocab.vocab_size

    def _load_data(self):
        """
        load processed data and encode, or load already encoded data
        """
        # directly read processed data and encode
        self.data = json.loads(open(self.cfg.data_path + self.cfg.data_file, 'r', encoding='utf-8').read().lower())
        self.train, self.dev, self.test = [], [], []
        for fn, dial in self.data.items():
            if '.json' in fn:
                fn = fn.replace('.json', '')

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx + 1]
                        ns = self.vocab.decode(ns) if type(ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx + 1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        vector = self.db.addDBIndicator(match_dom, match)
        return vector
