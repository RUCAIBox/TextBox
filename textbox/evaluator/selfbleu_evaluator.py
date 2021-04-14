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

ngram_metrics = {metric.lower(): metric for metric in ['self-bleu']}

class SelfBleuEvaluator():
    r"""Bleu Evaluator. Now, we support two ngram metrics which contains `'bleu'`.
    """

    def __init__(self, config):
        self.metrics = config['metrics']
        self.n_grams = config['n_grams']
        # [1, 2, 3, 4]
        self._check_args()
    
    def evaluate(self, generate_corpus, reference_corpus):
        """get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus
        
        Returns:
            dict: such as ``{'self-bleu-1': xxx, 'self-bleu-1-avg': yyy}``
        """
        # get metrics
        metric_dict = {}
        bleu_dict = self._calc_metrics_info(
            generate_corpus=generate_corpus, reference_corpus=reference_corpus
        )
        for n_gram in bleu_dict:
            key = 'self-bleu-{}'.format(n_gram)
            tp_list = bleu_dict[n_gram]
            tp_val = np.mean(tp_list)
            metric_dict[key] = round(tp_val, 4)
        return metric_dict
        

    def _check_args(self):
        self.metrics = ['self-bleu']

        # Check n_gram
        if isinstance(self.n_grams, (int, list)):
            if isinstance(self.n_grams, int):
                self.n_grams = [self.n_grams]
            for n_gram in self.n_grams:
                if n_gram <= 0:
                    raise ValueError("n_gram must be a positive integer or a list of positive integers, but get `{}`".format(n_gram))
        else:
            raise TypeError("The n_gram must be a integer, list")
    
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
            weights[str(n_gram)] = tuple(weight)
            weight[n_gram - 1] = 0.0
        for n_gram in self.n_grams:
            avg_weight = [1. / n_gram] * n_gram
            avg_weight.extend([0. for i in range(max(self.n_grams) - n_gram)])
            weights[str(n_gram) + "-avg"] = tuple(avg_weight)
        
        bleu = SelfBLEU(generate_corpus, weights)
        scores = bleu.get_score()

        results = []
        avg_results = []
        for n_gram in self.n_grams:
            score = np.array(scores[str(n_gram)])
            results.append(score.mean())
        for n_gram in self.n_grams:
            score = np.array(scores[str(n_gram) + "-avg"])
            avg_results.append(score.mean())
        return results, avg_results

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """

        bleu_dict = {}
        for i in self.n_grams:
            bleu_dict[str(i)] = []
        for i in self.n_grams:
            bleu_dict[str(i) + "-avg"] = []
        result, avg_result = self._self_bleu(generate_corpus=generate_corpus)
        for index in range(len(self.n_grams)):
            bleu_dict[str(self.n_grams[index])].append(result[index])
            bleu_dict[str(self.n_grams[index]) + "-avg"].append(avg_result[index])
        return bleu_dict
    
    def __str__(self):
        mesg = 'The Self-Bleu Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [ngram_metrics[metric.lower()] for metric in self.metrics]) \
               + '], Ngram:[' + ', '.join(map(str, self.n_grams)) + ']'
        return mesg