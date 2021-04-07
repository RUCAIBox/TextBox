# @Time   : 2021/3/23
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
textbox.evaluator.dialog_evaluator
#######################################
"""

import numpy as np
import torch
from textbox.evaluator.abstract_evaluator import AbstractEvaluator
from textbox.evaluator.metrics import metrics_dict


class DialogEvaluator(AbstractEvaluator):
    r"""Dialog Evaluator is mainly used in dialog tasks. Now, we support metrics `'bleu'` and `'distinct'`.
    """

    def __init__(self, config):
        super().__init__(config)

        self.n_grams = config['n_grams']

    def evaluate(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: such as ``{'bleu-1': xxx}``
        """
        # get metrics
        metric_dict = {}
        bleu_dict = self._calc_bleu_info(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
        for n_gram in bleu_dict:
            key = 'bleu-{}'.format(n_gram)
            tp_list = bleu_dict[n_gram]
            tp_val = np.mean(tp_list)
            metric_dict[key] = round(tp_val, 4)

        dist_dict = self._calc_dist_info(generate_corpus=generate_corpus)
        for key in dist_dict:
            tp_val = dist_dict[key]
            metric_dict[key] = round(tp_val, 4)
        return metric_dict

    def _calc_bleu_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            list: a list of metrics <metric> which record the results according to self.ngrams
        """
        metric_fuc = metrics_dict['bleu']
        assert len(generate_corpus) == len(reference_corpus)
        bleu_dict = {}
        for i in self.n_grams:
            bleu_dict[i] = []
        bleu_dict['avg-bleu'] = []
        for i in range(len(generate_corpus)):
            pred_sent = generate_corpus[i]
            gold_sent = reference_corpus[i]
            result, avg_bleu = metric_fuc(
                generate_corpus=[pred_sent], reference_corpus=[gold_sent], n_grams=self.n_grams, get_avg=True
            )
            for i in self.n_grams:
                bleu_dict[i].append(result[i - 1])
            bleu_dict['avg-bleu'].append(avg_bleu)
        return bleu_dict

    def dist_func(self, generate_corpus, ngram):
        ngram_pool = []
        for sent in generate_corpus:
            tokens = sent[:ngram]
            ngram_pool.append(' '.join(tokens))
            for i in range(1, len(sent) - ngram + 1):
                tokens = tokens[1:]
                tokens.append(sent[i + ngram - 1])
                ngram_pool.append(' '.join(tokens))
        return len(set(ngram_pool)) / len(ngram_pool)

    def _calc_dist_info(self, generate_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus

        Returns:
            list: a list of metrics <metric> which record the results
        """
        dist_dict = {}
        for i in range(1, 3):
            result = self.dist_func(generate_corpus, i)
            key = "distinct-{}".format(i)
            dist_dict[key] = result
        return dist_dict
