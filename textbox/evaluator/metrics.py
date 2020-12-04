# -*- encoding: utf-8 -*-
# @Time    :   2020/11/16
# @Author  :   Junyi Li
# @email   :   lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
recbole.evaluator.metrics
############################
"""

from logging import getLogger

import numpy as np
from fast_bleu import BLEU, SelfBLEU


def bleu_(generate_corpus, reference_corpus, n_grams):
    weight = [0] * max(n_grams)
    weights = {}
    for n_gram in n_grams:
        weight[n_gram - 1] = 1.0
        weights[n_gram] = tuple(weight)
        weight[n_gram - 1] = 0.0

    bleu = BLEU(reference_corpus, weights)
    scores = bleu.get_score(generate_corpus)

    results = []
    for n_gram in n_grams:
        score = np.array(scores[n_gram])
        results.append(score.mean())
    return results

def self_bleu_(generate_corpus, n_grams, reference_corpus=None):
    weight = [0] * max(n_grams)
    weights = {}
    for n_gram in n_grams:
        weight[n_gram - 1] = 1.0
        weights[n_gram] = tuple(weight)
        weight[n_gram - 1] = 0.0

    bleu = SelfBLEU(generate_corpus, weights)
    scores = bleu.get_score()

    results = []
    for n_gram in n_grams:
        score = np.array(scores[n_gram])
        results.append(score.mean())
    return results


"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'bleu': bleu_,
    'self_bleu': self_bleu_
}
