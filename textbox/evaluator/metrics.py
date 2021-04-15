# -*- encoding: utf-8 -*-
# @Time    :   2020/11/16
# @Author  :   Junyi Li
# @email   :   lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
textbox.evaluator.metrics
############################
"""

import numpy as np
from fast_bleu import BLEU, SelfBLEU


def bleu_(generate_corpus, reference_corpus, n_grams, get_avg=False):
    r""" Calculate the BLEU metrics of the generated corpus in referenced corpus.

    Args:
        generate_corpus (List[List[str]]): the generated corpus
        reference_corpus (List[List[str]]): the referenced corpus
        n_grams (List): the n-gram metric to be calculated
        get_avg (Bool, optional): whether to calculate the average BLEU score, default: False

    Returns:
        List or (List, float): the BLEU results and optinal average BLEU score
    """
    
    weight = [0] * max(n_grams)
    weights = {}
    for n_gram in n_grams:
        weight[n_gram - 1] = 1.0
        weights[n_gram] = tuple(weight)
        weight[n_gram - 1] = 0.0
    if get_avg:
        weights['avg-bleu'] = tuple([0.25] * 4)

    bleu = BLEU(reference_corpus, weights)
    scores = bleu.get_score(generate_corpus)

    results = []
    for n_gram in n_grams:
        score = np.array(scores[n_gram])
        results.append(score.mean())
    if get_avg:
        avg_bleu = np.array(scores['avg-bleu']).mean()
        return results, avg_bleu
    return results


def self_bleu_(generate_corpus, n_grams, reference_corpus=None):
    r""" Calculate the Self-BLEU metrics of the generated corpus.

    Args:
        generate_corpus (List[List[str]]): the generated corpus
        n_grams (List): the n-gram metric to be calculated

    Returns:
        List: the Self-BLEU results
    """

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
metrics_dict = {'bleu': bleu_, 'self_bleu': self_bleu_}
