import numpy as np
from fast_bleu import BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textbox.evaluator.abstract_evaluator import AbstractEvaluator


class BleuEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, task_type):
        self.n_grams = [1, 2, 3, 4]
        self.task_type = task_type
        self.weights = self._generate_weights()

    def _generate_weights(self):
        weight = [0] * max(self.n_grams)
        weights = {}
        for n_gram in self.n_grams:
            weight[n_gram - 1] = 1.0
            weights['bleu-{}'.format(n_gram)] = tuple(weight)
            weight[n_gram - 1] = 0.0
            avg_weight = [1. / n_gram] * n_gram
            avg_weight.extend([0. for index in range(max(self.n_grams) - n_gram)])
            weights['bleu-{}-avg'.format(n_gram)] = tuple(avg_weight)
        return weights

    def _calc_fast_bleu(self, generate_corpus, reference_corpus):
        r""" Calculate the BLEU metrics of the generated corpus in referenced corpus.

        Args:
            generate_corpus (List[List[str]]): the generated corpus
            reference_corpus (List[List[str]]): the referenced corpus
            n_grams (List): the n-gram metric to be calculated

        Returns:
            list: the BLEU results and average BLEU scores
        """

        bleu = BLEU(reference_corpus, self.weights)
        scores = bleu.get_score(generate_corpus)
        return scores

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.n_grams
        """

        bleu_dict = {}
        for n_gram in self.n_grams:
            bleu_dict['bleu-{}'.format(n_gram)] = []
        for n_gram in self.n_grams:
            bleu_dict['bleu-{}-avg'.format(n_gram)] = []

        if self.task_type:
            results = self._calc_fast_bleu(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
            for n_gram in self.n_grams:
                bleu_dict['bleu-{}'.format(n_gram)].append(np.array(results['bleu-{}'.format(n_gram)]).mean())
                bleu_dict['bleu-{}-avg'.format(n_gram)].append(np.array(results['bleu-{}-avg'.format(n_gram)]).mean())
        else:
            for i in range(len(generate_corpus)):
                pred_sent = generate_corpus[i]
                gold_sent = reference_corpus[i]
                results = sentence_bleu(
                    hypothesis=pred_sent,
                    references=[gold_sent],
                    weights=self.weights,
                    smoothing_function=SmoothingFunction().method1
                )
                for n_gram in self.n_grams:
                    bleu_dict['bleu-{}'.format(n_gram)].append(np.array(results['bleu-{}'.format(n_gram)]).mean())
                    bleu_dict['bleu-{}-avg'.format(n_gram)].append(
                        np.array(results['bleu-{}-avg'.format(n_gram)]).mean()
                    )
        return bleu_dict
