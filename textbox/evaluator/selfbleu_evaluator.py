# @Time   : 2020/11/14
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

# UPDATE
# @Time   : 2021/4/12
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.selfbleu_evaluator
#######################################
"""
import numpy as np
from fast_bleu import SelfBLEU
from textbox.evaluator.abstract_evaluator import AbstractEvaluator

class SelfBleuEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'self-bleu'`.
    """

    def __init__(self):
        self.n_grams = [1, 2, 3, 4]
    
    def _self_bleu(self, generate_corpus):
        r""" Calculate the Self-BLEU metrics of the generated corpus in referenced corpus.

        Args:
            generate_corpus (List[List[str]]): the generated corpus
            n_grams (List): the n-gram metric to be calculated

        Returns:
            list: the Self-BLEU results
        """

        weight = [0] * max(self.n_grams)
        weights = {}
        for n_gram in self.n_grams:
            weight[n_gram - 1] = 1.0
            weights["self-bleu-{}".format(n_gram)] = tuple(weight)
            weight[n_gram - 1] = 0.0
            avg_weight = [1. / n_gram] * n_gram
            avg_weight.extend([0. for i in range(max(self.n_grams) - n_gram)])
            weights["self-bleu-{}-avg".format(n_gram)] = tuple(avg_weight)
        
        bleu = SelfBLEU(generate_corpus, weights)
        scores = bleu.get_score()

        results = {}
        for n_gram in self.n_grams:
            score = np.array(scores['self-bleu-{}'.format(n_gram)])
            results['self-bleu-{}'.format(n_gram)] = score.mean()
        for n_gram in self.n_grams:
            score = np.array(scores['self-bleu-{}-avg'.format(n_gram)])
            results['self-bleu-{}-avg'.format(n_gram)] = score.mean()
        return results

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.n_grams
        """

        bleu_dict = {}
        for n_gram in self.n_grams:
            bleu_dict['self-bleu-{}'.format(n_gram)] = []
        for n_gram in self.n_grams:
            bleu_dict['self-bleu-{}-avg'.format(n_gram)] = []
        
        results = self._self_bleu(generate_corpus=generate_corpus)
        for n_gram in self.n_grams:
            bleu_dict['self-bleu-{}'.format(n_gram)].append(results['self-bleu-{}'.format(n_gram)])
            bleu_dict['self-bleu-{}-avg'.format(n_gram)].append(results['self-bleu-{}-avg'.format(n_gram)])
        return bleu_dict
    
    def __str__(self):
        mesg = 'The Self-Bleu Evaluator Info:\n' + '\tMetrics:[self-bleu], Ngram:[' + ', '.join(map(str, self.n_grams)) + ']'
        return mesg
