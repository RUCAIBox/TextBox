# -*- encoding: utf-8 -*-
# @Time    :   2020/11/16
# @Author  :   Junyi Li
# @email   :   lijunyi@ruc.edu.cn

"""
recbole.evaluator.metrics
############################
"""

from logging import getLogger

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def bleu_(generate_corpus, reference_corpus, n_gram):
    weights = [0, 0, 0, 0, 0]
    weights[n_gram-1] = 1
    weights = tuple(weights)
    bleu_score = []
    for candidate in generate_corpus:
        bleu_score.append(sentence_bleu(reference_corpus, candidate, weights,
                                        smoothing_function=SmoothingFunction().method1))
    return sum(bleu_score) / len(bleu_score)


def self_bleu_(generate_corpus, n_gram, reference_corpus=None):
    weights = [0, 0, 0, 0, 0]
    weights[n_gram-1] = 1
    weights = tuple(weights)
    self_bleu_score = []
    for idx in range(len(generate_corpus)):
        candidate = generate_corpus[idx]
        reference_corpus = generate_corpus[:idx] + generate_corpus[idx+1:]
        self_bleu_score.append(sentence_bleu(reference_corpus, candidate, weights,
                                             smoothing_function=SmoothingFunction().method1))
    return sum(self_bleu_score) / len(self_bleu_score)


"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'bleu': bleu_,
    'self_bleu': self_bleu_
}
