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
ngram_metrics = {metric.lower(): metric for metric in ['bleu']}


class TranslationEvaluator(AbstractEvaluator):
    r"""NgramEvaluator Evaluator is mainly used in ranking tasks. Now, we support two ngram metrics which
    contain `'bleu', 'self_bleu'.
    """

    def __init__(self, config):
        super().__init__(config)

        self.n_grams = config['n_grams']
        # [1, 2, 3, 4]
        # config['n_grams']
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
        # result_dict = self._calculate_metrics(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
        bleu_dict = self._calc_metrics_info(generate_corpus=generate_corpus,
                                            reference_corpus=reference_corpus,
                                            metric='bleu')
        bleu_sum = 0.0
        for n_gram in bleu_dict:
            key = 'bleu-{}'.format(n_gram)
            tp_list = bleu_dict[n_gram]
            tp_val = np.mean(tp_list)
            metric_dict[key] = round(tp_val, 4)
            bleu_sum += tp_val
        avg_bleu = bleu_sum / 4.0
        metric_dict['avg-bleu'] = round(avg_bleu, 4)
        return metric_dict

    def _check_args(self):
        self.metrics = ['bleu']

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

    def _calc_metrics_info(self, generate_corpus, reference_corpus, metric="bleu"):
        """get metrics result
        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            list: a list of metrics <metric> which record the results according to self.ngrams

        """
        assert metric in metrics_dict
        metric_fuc = metrics_dict[metric.lower()]
        assert len(generate_corpus) == len(reference_corpus)
        bleu_dict = {}
        for i in self.n_grams:
            bleu_dict[i] = []
        for i in range(len(generate_corpus)):
            pred_sent = generate_corpus[i]
            gold_sent = reference_corpus[i]
            result = metric_fuc(generate_corpus=[pred_sent], reference_corpus=[gold_sent], n_grams=self.n_grams)
            for i in self.n_grams:
                bleu_dict[i].append(result[i-1])
        return bleu_dict

    def __str__(self):
        mesg = 'The Translation Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [ngram_metrics[metric.lower()] for metric in self.metrics]) \
               + '], Ngram:[' + ', '.join(map(str, self.n_grams)) + ']'
        return mesg
