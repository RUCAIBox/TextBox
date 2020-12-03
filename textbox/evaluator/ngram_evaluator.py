# @Time   : 2020/11/14
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
textbox.evaluator.ngram_evaluator
################################
"""

import numpy as np
import torch
from textbox.evaluator.abstract_evaluator import AbstractEvaluator
from textbox.evaluator.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence

# These metrics are typical in topk recommendations
ngram_metrics = {metric.lower(): metric for metric in ['bleu', 'self_bleu']}


class NgramEvaluator(AbstractEvaluator):
    r"""NgramEvaluator Evaluator is mainly used in ranking tasks. Now, we support two ngram metrics which
    contain `'bleu', 'self_bleu'.
    """

    def __init__(self, config):
        super().__init__(config)

        self.n_grams = config['n_grams']
        # [1, 2, 3, 4]
        self._check_args()

    def evaluate(self, generate_corpus, reference_corpus):
        """get metrics result
        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: such as ``{'bleu-1': xxx, 'self-bleu-4': yyyy}``
        """
        # assert len(generate_corpus) == len(reference_corpus)
        # get metrics
        metric_dict = {}
        result_dict = self._calculate_metrics(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
        for metric in result_dict:
            result_list = result_dict[metric]
            for index, n_gram in enumerate(self.n_grams):
                key = '{}-{}'.format(metric, n_gram)
                metric_dict[key] = round(result_list[index], 4)
                # reserve float number as .1234
        return metric_dict

    def _check_args(self):

        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in ngram_metrics:
                raise ValueError("There is no user grouped ngram metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.n_grams, (int, list)):
            if isinstance(self.n_grams, int):
                self.n_grams = [self.n_grams]
            for n_gram in self.n_grams:
                if n_gram <= 0:
                    raise ValueError(
                        'n_gram must be a positive integer or a list of positive integers, but get `{}`'.format(n_gram))
        else:
            raise TypeError('The n_gram must be a integer, list')

    def metrics_info(self, generate_corpus, reference_corpus, metric="bleu"):
        """get metrics result
        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            list: a list of metrics <metric> which record the results according to self.ngrams

        """
        assert metric in metrics_dict
        metric_fuc = metrics_dict[metric.lower()]
        result_list = []
        result = metric_fuc(generate_corpus=generate_corpus, reference_corpus=reference_corpus, n_grams=self.n_grams)
        result_list.extend(result)
        return result_list

    def _calculate_metrics(self, generate_corpus, reference_corpus):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users

        Returns:
            np.ndarray: a matrix which contains the metrics result

        """
        result_dict = {}
        for metric in self.metrics:
            result_list = self.metrics_info(generate_corpus, reference_corpus, metric)
            result_dict[metric] = result_list
        return result_dict

    def __str__(self):
        mesg = 'The Ngram Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [ngram_metrics[metric.lower()] for metric in self.metrics]) \
               + '], Ngram:[' + ', '.join(map(str, self.n_grams)) + ']'
        return mesg
