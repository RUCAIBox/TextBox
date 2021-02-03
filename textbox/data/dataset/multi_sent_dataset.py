# @Time   : 2021/2/3
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.multi_sent_dataset
########################################
"""

import os
from textbox.data.dataset import AbstractDataset
from textbox.data.utils import tokenize, split_data, build_vocab, detect_restored, dump_data, load_restored


class MultipleSentenceDataset(AbstractDataset):

    def __init__(self, config):
        self.language = config['language'].lower()
        self.max_vocab_size = config['max_vocab_size']
        self.max_utt_length = config['max_utt_length']
        self.max_ctx_turn = config['max_ctx_turn']
        self.max_knowledge_num = config['max_knowledge_num']
        self.max_knowledge_len = config['max_knowledge_len']
        super().__init__(config)

    def __len__(self):
        return sum([len(data) for data in self.text_data])

    def _get_preset(self):
        self.token2idx = {}
        self.idx2token = {}
        self.knowledge_text_data = []
        self.source_text_data = []
        self.target_text_data = []

    def _load_multi_data(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise ValueError('File {} not exist'.format(dataset_path))
    
        fin = open(dataset_path, "r")
        knowledge_text = []
        source_text = []
        target_text = []
        for line in fin:
            knowledge, src, tgt = line.strip().lower().split('\t')
            tgt = tokenize(tgt, self.tokenize_strategy, self.language)
            knowledge = [tokenize(k, self.tokenize_strategy, self.language) for k in knowledge.split(" __eou__ ")]
            knowledge = [k[:self.max_knowledge_len] for k in knowledge[-self.max_knowledge_num:]]
            src = [tokenize(s, self.tokenize_strategy, self.language) for s in src.split(" __eou__ ")]
            if self.overlength_strategy == 'drop':
                if len(src) > self.max_ctx_turn or len(tgt) > self.max_utt_length or any([len(s) > self.max_utt_length for s in src]):
                    continue
            else:
                src = [s[:self.max_utt_length] for s in src[-self.max_ctx_turn:]]
                tgt = tgt[:self.max_utt_length]
            
            knowledge_text.append(knowledge)
            source_text.append(src)
            target_text.append(tgt)
        return knowledge_text, source_text, target_text

    def _load_split_data(self, dataset_path):
        """Load dataset from split (train, dev, test).
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        for prefix in ['train', 'dev', 'test']:
            filename = os.path.join(dataset_path, '{}.txt'.format(prefix))
            knowledge_text_data, source_text_data, target_text_data = self._load_multi_data(filename)
            self.knowledge_text_data.append(knowledge_text_data)
            self.source_text_data.append(source_text_data)
            self.target_text_data.append(target_text_data)

    def _load_single_data(self, dataset_path):
        """Load full corpus.
        This is designed for single sentence format, unconditional task.
        Args:
            dataset_path (str): path of dataset dir.
        """
        dataset_file = os.path.join(dataset_path, 'corpus.txt')
        knowledge_text_data, source_text_data, target_text_data = self._load_multi_data(dataset_file)
        self.knowledge_text_data, self.source_text_data, self.target_text_data = split_data([knowledge_text_data, source_text_data, target_text_data], self.split_ratio)

    def _load_data(self, dataset_path):
        if self.split_strategy == "load_split":
            self._load_split_data(dataset_path)
        elif self.split_strategy == "by_ratio":
            self._load_single_data(dataset_path)
        else:
            raise NotImplementedError("{} split strategy not implemented".format(self.split_strategy))

    def _build_vocab(self):
        text_data = self.knowledge_text_data + self.source_text_data + self.target_text_data
        self.idx2token, self.token2idx, self.max_vocab_size = build_vocab(text_data, self.max_vocab_size, self.special_token_list)

    def _detect_restored(self, dataset_path):
        return detect_restored(dataset_path, 'knowledge.', ignore_vocab=True) and detect_restored(dataset_path, 'source.') and detect_restored(dataset_path, 'target.', ignore_vocab=True)

    def _dump_data(self, dataset_path):
        dump_data(dataset_path, self.knowledge_text_data, suffix='knowledge.')
        dump_data(dataset_path, self.source_text_data, self.idx2token, self.token2idx, 'source.')
        dump_data(dataset_path, self.target_text_data, suffix='target.')
        self.logger.info("Dump finished!")

    def _load_restored(self, dataset_path):
        """Load dataset from restored binary files (train, dev, test).
        Args:
            dataset_path (str): path of dataset dir.
        """
        self.knowledge_text_data = load_restored(dataset_path, 'knowledge.', ignore_vocab=True)
        self.source_text_data, self.idx2token, self.token2idx = load_restored(dataset_path, 'source.')
        self.target_text_data = load_restored(dataset_path, 'target.', ignore_vocab=True)
        self._max_vocab_size = len(self.idx2token)
        self.logger.info("Restore finished!")

    def build(self):
        info_str = ''
        corpus_list = []
        self.logger.info(
            "Vocab size: {}".format(self.max_vocab_size)
        )

        for i, prefix in enumerate(['train', 'dev', 'test']):
            knowledge_text_data = self.knowledge_text_data[i]
            source_text_data = self.source_text_data[i]
            target_text_data = self.target_text_data[i]
            tp_data = {
                'idx2token': self.idx2token,
                'token2idx': self.token2idx,
                'knowledge_text_data': knowledge_text_data,
                'source_text_data': source_text_data,
                'target_text_data': target_text_data
            }
            corpus_list.append(tp_data)
            info_str += '{}: {} cases, '.format(prefix, len(source_text_data))
        
        self.logger.info(info_str[:-2] + '\n')
        return corpus_list
