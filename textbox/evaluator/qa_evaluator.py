import string
import re
import numpy as np
from .abstract_evaluator import AbstractEvaluator
from collections import Counter


class QaEvaluator(AbstractEvaluator):
    r"""Qa Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        super(QaEvaluator, self).__init__(config)
        self.multiref_strategy = config['multiref_strategy']

    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(s.lower())))

    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _exact_match_score(self, prediction, ground_truth):
        return (self._normalize_answer(prediction) == self._normalize_answer(ground_truth))

    def _metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        if len(scores_for_ground_truths) > 1 and self.multiref_strategy == 'leave_one_out':
            func = lambda x: (max(x) * (len(x) - 1) + np.partition(x, -2)[-2]) / len(x)
        else:
            func = max
        return func(scores_for_ground_truths)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        results = {'em': [], 'f1': []}

        for gen, refs in zip(generate_corpus.tokenized_text, reference_corpus.tokenized_text):
            em = self._metric_max_over_ground_truths(self._exact_match_score, gen, refs)
            f1 = self._metric_max_over_ground_truths(self._f1_score, gen, refs)
            results['em'].append(em * 100)
            results['f1'].append(f1 * 100)

        return results
