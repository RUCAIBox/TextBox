# @Time   : 2020/11/16
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2021/10/10, 2021/1/29, 2021/10/9
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

"""
textbox.data.dataset.abstract_dataset
#####################################
"""

import os
import torch
from logging import getLogger
from textbox.utils.enum_type import SpecialTokens
from textbox.data.utils import load_data


class AbstractDataset(object):
    """:class:`AbstractDataset` is an abstract object which stores the original dataset in memory.
        And it is also the ancestor of all other dataset.

    Args:
        config (Config): Global configuration object.
    """

    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path']
        self.source_language = (config['src_lang'] or 'english').lower()
        self.target_language = (config['tgt_lang'] or 'english').lower()
        self.source_vocab_size = int(config['src_vocab_size'] or config['vocab_size'] or 1e8)
        self.target_vocab_size = int(config['tgt_vocab_size'] or config['vocab_size'] or 1e8)
        self.source_max_length = int(config['src_len'] or config['seq_len'] or 1e4)
        self.target_max_length = int(config['tgt_len'] or config['seq_len'] or 1e4)
        self.source_multi_sentence = config['src_multi_sent'] or False
        self.target_multi_sentence = config['tgt_multi_sent'] or False
        self.source_max_num = int(config['src_num'] or 1e4)
        self.target_max_num = int(config['tgt_num'] or 1e4)
        self.tokenize_strategy = config['tokenize_strategy'] or 'by_space'

        self.logger = getLogger()
        self._init_special_token()
        self._get_preset()
        self.restored_exist = self._detect_restored()
        if self.restored_exist:
            self._from_restored()
        else:
            self._from_scratch()
        self._info()

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        for prefix in ['train', 'valid', 'test']:
            setattr(self, f'{prefix}_data', dict())

    def _init_special_token(self):
        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS
        self.padding_token_idx = 0
        self.unknown_token_idx = 1
        self.sos_token_idx = 2
        self.eos_token_idx = 3
        self.special_token_list = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token]
        if 'user_token_list' in self.config:
            self.user_token_list = self.config['user_token_list']
            self.special_token_list += self.user_token_list
            self.user_token_idx = [4 + i for i, _ in enumerate(self.user_token_list)]

    def _from_scratch(self):
        """Load dataset from scratch. Firstly load data from atomic files, then build vocabulary, dump data lastly.
        """
        self.logger.info('Loading data from scratch')
        self._load_target_data()
        self._load_source_data()
        if self.tokenize_strategy != 'none':
            self._build_vocab()
            self._text2idx()
            if self.config['vocab_size'] is not None or self.source_vocab_size == 1e8:
                self.vocab_size = self.target_vocab_size
                self.idx2token = self.target_idx2token
                self.token2idx = self.target_token2idx
            if self.config['seq_len'] is not None or self.source_max_length == 1e4:
                self.max_length = self.target_max_length
        self._build_data()
        self._dump_data()

    def _from_restored(self):
        """Load dataset from restored binary files.
        """
        self.logger.info('Loading data from restored')
        self._load_restored()

    def _load_source_data(self):
        r"""Load dataset from source file (train, valid, test).
        """
        raise NotImplementedError('Method [_load_source_data] should be implemented.')

    def _build_vocab(self):
        r"""Build the vocabulary of text data.
        """
        raise NotImplementedError('Method [_build_vocab] should be implemented.')

    def _text2idx(self):
        r"""Map each token into idx.
        """
        raise NotImplementedError('Method [_text2idx] should be implemented.')

    def _build_data(self):
        r"""Prepare splitted data elements for dataloader.
        """
        for key, value in self.__dict__.items():
            if key.startswith(('source', 'target')) or key in ['vocab_size', 'max_length', 'idx2token', 'token2idx']:
                if isinstance(value, list) and isinstance(value[0], (list, str, int)) and len(value) in [2, 3]:
                    for i, (prefix, v) in enumerate(zip(['train', 'valid', 'test'], value)):
                        getattr(self, f'{prefix}_data')[key] = v
                else:
                    for prefix in ['train', 'valid', 'test']:
                        getattr(self, f'{prefix}_data')[key] = value

    def _load_target_data(self):
        """Load dataset from target file (train, valid, test).
        This is designed for single sentence format.
        """
        for prefix in ['train', 'valid', 'test']:
            filename = os.path.join(self.dataset_path, f'{prefix}.tgt')
            text_data = load_data(
                filename, self.tokenize_strategy, self.target_max_length, self.target_language,
                self.target_multi_sentence, self.target_max_num
            )
            self.target_text.append(text_data)

    def _detect_restored(self):
        r"""Detect whether binary files is already restored.

        Returns:
            bool: whether files are already restored.
        """
        absent_file_flag = False
        for prefix in ['train', 'valid', 'test']:
            filename = os.path.join(self.dataset_path, f'{prefix}.bin')
            if not os.path.isfile(filename):
                absent_file_flag = True
                break
        return not absent_file_flag

    def _dump_data(self):
        r"""dump dataset with processed dataset.
        """
        for prefix in ['train', 'valid', 'test']:
            filename = os.path.join(self.dataset_path, f'{prefix}.bin')
            data = getattr(self, f'{prefix}_data')
            torch.save(data, filename)

    def _load_restored(self):
        """Load dataset from restored binary files (train, valid, test).
        """
        for prefix in ['train', 'valid', 'test']:
            filename = os.path.join(self.dataset_path, f'{prefix}.bin')
            data = torch.load(filename)
            setattr(self, f'{prefix}_data', data)

        for key, value in self.test_data.items():
            if not isinstance(value, list):
                setattr(self, key, value)

    def _info(self):
        """Print the basic infomation of dataset.
        """
        info_str = ''
        self.logger.info("Vocab size: source {}, target {}".format(self.source_vocab_size, self.target_vocab_size))
        for prefix in ['train', 'valid', 'test']:
            data = getattr(self, f'{prefix}_data')['target_text']
            info_str += f'{prefix}: {len(data)} cases, '
        self.logger.info(info_str[:-2] + '\n')
