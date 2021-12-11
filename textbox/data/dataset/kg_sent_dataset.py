# @Time   : 2021/10/12
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.kg_sent_dataset
########################################
"""
import os

from textbox.data.dataset import AbstractDataset
from textbox.data.utils import build_vocab, text2idx


class KGSentenceDataset(AbstractDataset):

    def __init__(self, config):
        super().__init__(config)

    def _get_preset(self):
        super()._get_preset()
        self.source_text = []
        self.source_triple = []
        self.source_entity = []
        self.target_text = []
        self.target_mention = []
        self.relation = []
        self.target_dict = []

    def _load_kg_data(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise ValueError('File {} not exist'.format(dataset_path))

        text_data = []
        entity_data = []
        mention_data = []
        triple_data = []
        relation_data = []
        dict_data = []
        with open(dataset_path, "r") as fin:
            for line in fin:
                line = line.lower()
                line = line.strip().split('\t')
                text_data.append(line[0].split(' '))
                entity_data.append(line[1].split('|'))
                mention_data.append(line[2].split(' '))
                dict_data.append(line[3].split('|'))
                triple = []
                for tri in line[4:]:
                    head, relation, tail = tri.split('|')
                    triple.append((head.split(' '), relation, tail.split(' ')))
                    relation_data.append(relation)
                triple_data.append(triple)
        return text_data, entity_data, mention_data, triple_data, relation_data, dict_data

    def _load_source_data(self):
        for i, prefix in enumerate(['train', 'valid', 'test']):
            filename = os.path.join(self.dataset_path, f'{prefix}.src')
            text_data, entity_data, mention_data, triple_data, relation_data, dict_data = self._load_kg_data(filename)
            assert len(text_data) == len(self.target_text[i])
            self.source_text.append(text_data)
            self.source_entity.append(entity_data)
            self.target_mention.append(mention_data)
            self.source_triple.append(triple_data)
            self.relation.append(relation_data)
            self.target_dict.append(dict_data)

    def _build_vocab(self):
        data = self.source_text + self.target_text
        entity_data = [[[entity.split() for entity in doc] for doc in group] for group in self.source_entity]
        self.source_entity_idx2token, self.source_entity_token2idx, self.entity_vocab_size = build_vocab(
            entity_data, self.source_vocab_size, []
        )
        self.target_idx2token, self.target_token2idx, self.target_vocab_size = build_vocab(
            self.target_text, self.target_vocab_size, self.special_token_list
        )
        self.source_idx2token, self.source_token2idx, self.source_vocab_size = build_vocab(
            self.source_text, self.source_vocab_size, self.special_token_list
        )
        self.source_relation_idx2token, self.source_relation_token2idx, _ = build_vocab(
            self.relation, self.source_vocab_size, []
        )

    def _text2idx(self):
        self.source_idx, self.source_length, _ = text2idx(
            self.source_text, self.source_token2idx, self.tokenize_strategy
        )
        self.source_triple_idx = [[[([self.source_entity_token2idx[x] for x in h], self.source_relation_token2idx[r],
                                     [self.source_entity_token2idx[x] for x in t]) for h, r, t in doc] for doc in group]
                                  for group in self.source_triple]
        self.target_idx, self.target_length, _ = text2idx(
            self.target_text, self.target_token2idx, self.tokenize_strategy
        )
