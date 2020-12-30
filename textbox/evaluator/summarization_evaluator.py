# @Time   : 2020/11/14
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
textbox.evaluator.ngram_evaluator
#################################
"""

import numpy as np
import torch
from textbox.evaluator.abstract_evaluator import AbstractEvaluator
from textbox.evaluator.metrics import metrics_dict
import rouge


summarization_metrics = ['rouge-1', 'rouge-2', 'rouge-l', 'rouge-w']


class SummarizationEvaluator(AbstractEvaluator):
    r"""Summarization Evaluator is mainly used in summarization tasks. Now, we support rouge-based ngram metrics which
    contain rouge-n, rouge-l and rouge-w.
    """

    def __init__(self, config):
        super().__init__(config)

        self.n_grams = config['n_grams']
        self.evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                     max_n=2,
                                     limit_length=True,
                                     length_limit=config['target_max_seq_length'],
                                     length_limit_type='words',
                                     apply_avg=True,
                                     apply_best=False,
                                     alpha=0.5,  # Default F1_score
                                     weight_factor=1.2,
                                     stemming=True)

    def transform_words2str(self, corpus):
        new_corpus = []
        for words in corpus:
            new_corpus.append(" ".join(words))
        return new_corpus

    def evaluate(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: such as ``{'bleu-1': xxx, 'self-bleu-4': yyyy}``
        """
        assert len(generate_corpus) == len(reference_corpus)
        generate_corpus = self.transform_words2str(generate_corpus)
        reference_corpus = self.transform_words2str(reference_corpus)
        metric_dict = {}
        rouge_dict = self._calc_metrics_info(generate_corpus=generate_corpus,
                                             reference_corpus=reference_corpus)
        for metric in rouge_dict:
            tp_list = rouge_dict[metric]
            tp_val = np.mean(tp_list)
            metric_dict[metric] = round(tp_val, 4)
        return metric_dict

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            list: a list of metrics <metric> which record the results according to self.ngrams
        """
        assert len(generate_corpus) == len(reference_corpus)
        rouge_dict = {}
        for i in summarization_metrics:
            rouge_dict[i] = []
        for i in range(len(generate_corpus)):
            pred_sent = generate_corpus[i]
            gold_sent = reference_corpus[i]
            result = self.calc_rouge(gen_corpus=[pred_sent], ref_corpus=[gold_sent])
            for key_ in rouge_dict:
                rouge_dict[key_].append(result[key_]['f'])
        return rouge_dict

    def calc_rouge(self, gen_corpus, ref_corpus):
        scores = self.evaluator.get_scores(gen_corpus, ref_corpus)
        return scores

    def __str__(self):
        mesg = 'The Summarization Evaluator Info:\n' + '\tMetrics:[' + ', '.join(summarization_metrics) + ']',
        return mesg
