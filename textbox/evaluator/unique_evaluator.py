# @Time   : 2021/5/1
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.unique_evaluator
#######################################
"""

import numpy as np
from nltk.util import ngrams
from collections import Counter, defaultdict
from textbox.evaluator.abstract_evaluator import AbstractEvaluator


class UniqueEvaluator(AbstractEvaluator):
    r"""Unique Evaluator. Now, we support metrics `'unique'`.
    """

    def __init__(self):
        self.n_grams = [1, 2, 3, 4]

    def _generate_ngrams(self, input_sentence, ngram):
        ngram_dict = defaultdict(int)
        tokens = input_sentence[:ngram]
        ngram_dict[" ".join(tokens)] += 1
        for i in range(1, len(input_sentence) - ngram + 1):
            tokens = tokens[1:]
            tokens.append(input_sentence[i + ngram - 1])
            ngram_dict[" ".join(tokens)] += 1
        return ngram_dict

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        result = {}
        for ngram in self.n_grams:
            ngram_all = Counter()
            for generate_sentence in generate_corpus:
                ngram_dict = self._generate_ngrams(generate_sentence, ngram)
                ngram_all.update(ngram_dict)
            result['unique-{}'.format(ngram)
                   ] = sum(filter(lambda x: x == 1, ngram_all.values())) / sum(ngram_all.values())
        return result
