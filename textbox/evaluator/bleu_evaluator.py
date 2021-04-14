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


# 然后n-gram这个可以去掉
# 算bleu1234和avg-bleu1234     BleuEvaluator
# rouge 算1,2,L,W     RougeEvaluator
# distinct算1234     DistinctEvaluator
# 然后还有一个nll_test（这个在evaluate里面直接算）

# weights = {'1': (1.0, 0.0, 0.0, 0.0), '2': (0.0, 1.0, 0.0, 0.0),
#            '3': (0.0, 0.0, 1.0, 0.0), '4': (0.0, 0.0, 0.0, 1.0), 
#            '1a': (1.0, 0.0, 0.0, 0.0), '2a': (0.5, 0.5, 0.0, 0.0),
#            '3a': (1./3, 1./3, 1./3, 0.0), '4a': (0.25, 0.25, 0.25, 0.25)}
"""
textbox.evaluator.bleu_evaluator
#######################################
"""

import numpy as np
from fast_bleu import BLEU

ngram_metrics = {metric.lower(): metric for metric in ['bleu']}

class BleuEvaluator():
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
            dict: such as ``{'bleu-1': xxx, 'bleu-1-avg': yyy}``
        """
        # get metrics
        metric_dict = {}
        bleu_dict = self._calc_metrics_info(
            generate_corpus=generate_corpus, reference_corpus=reference_corpus
        )
        for n_gram in bleu_dict:
            key = 'bleu-{}'.format(n_gram)
            tp_list = bleu_dict[n_gram]
            tp_val = np.mean(tp_list)
            metric_dict[key] = round(tp_val, 4)
        return metric_dict
        

    def _check_args(self):
        self.metrics = ['bleu']

        # Check n_gram
        if isinstance(self.n_grams, (int, list)):
            if isinstance(self.n_grams, int):
                self.n_grams = [self.n_grams]
            for n_gram in self.n_grams:
                if n_gram <= 0:
                    raise ValueError("n_gram must be a positive integer or a list of positive integers, but get `{}`".format(n_gram))
        else:
            raise TypeError("The n_gram must be a integer, list")
    
    def _bleu(self, generate_corpus, reference_corpus):
        r""" Calculate the BLEU metrics of the generated corpus in referenced corpus.

        Args:
            generate_corpus (List[List[str]]): the generated corpus
            reference_corpus (List[List[str]]): the referenced corpus
            n_grams (List): the n-gram metric to be calculated

        Returns:
            List or (List, float): the BLEU results and average BLEU scores
        """

        weight = [0] * max(self.n_grams)
        weights = {}
        for n_gram in self.n_grams:
            weight[n_gram - 1] = 1.0
            weights[str(n_gram)] = tuple(weight)
            weight[n_gram - 1] = 0.0
        for n_gram in self.n_grams:
            avg_weight = [1. / n_gram] * n_gram
            avg_weight.extend([0. for index in range(max(self.n_grams) - n_gram)])
            weights[str(n_gram) + "-avg"] = tuple(avg_weight)
        
        bleu = BLEU(reference_corpus, weights)
        scores = bleu.get_score(generate_corpus)

        results = []
        avg_results = []
        for n_gram in self.n_grams:
            score = np.array(scores[str(n_gram)])
            results.append(score.mean())
        for n_gram in self.n_grams:
            score = np.array(scores[str(n_gram) + "-avg"])
            avg_results.append(score.mean())

        return results, avg_results

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """

        assert len(generate_corpus) == len(reference_corpus)
        bleu_dict = {}
        for i in self.n_grams:
            bleu_dict[str(i)] = []
        for i in self.n_grams:
            bleu_dict[str(i) + "-avg"] = []
        results, avg_results = self._bleu(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
        for index in range(len(self.n_grams)):
            bleu_dict[str(self.n_grams[index])].append(results[index])
            bleu_dict[str(self.n_grams[index]) + "-avg"].append(avg_results[index])
        # for i in range(len(generate_corpus)):
        #     pred_sent = generate_corpus[i]
        #     gold_sent = reference_corpus[i]
        #     results, avg_results = self._bleu(
        #         generate_corpus=[pred_sent], reference_corpus=[gold_sent]
        #     )
        #     print(results, avg_results)
        #     break
        #     for index in range(len(self.n_grams)):
        #         bleu_dict[str(self.n_grams[index])].append(results[index])
        #         bleu_dict[str(self.n_grams[index]) + "-avg"].append(avg_results[index])
        return bleu_dict
    
    def __str__(self):
        mesg = 'The Bleu Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [ngram_metrics[metric.lower()] for metric in self.metrics]) \
               + '], Ngram:[' + ', '.join(map(str, self.n_grams)) + ']'
        return mesg
