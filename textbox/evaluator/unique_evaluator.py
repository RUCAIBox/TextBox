from collections import Counter
from .abstract_evaluator import AbstractEvaluator


class UniqueEvaluator(AbstractEvaluator):
    r"""Unique Evaluator. Now, we support metrics `'unique'`.
    """

    def __init__(self, config):
        super(UniqueEvaluator, self).__init__(config)
        self.unique_max_ngrams = config['unique_max_ngrams']

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        results = {}
        ngrams_all = [Counter() for _ in range(self.unique_max_ngrams)]

        for gen in generate_corpus.tokens:
            ngrams = []
            for i in range(self.unique_max_ngrams):
                ngrams.append(gen[i:])
                ngram = Counter(zip(*ngrams))
                ngrams_all[i].update(ngram)

        for i in range(self.unique_max_ngrams):
            results[f'unique-{i+1}'] = sum(filter(lambda x: x == 1, ngrams_all[i].values())
                                           ) / sum(ngrams_all[i].values()) * 100

        return results
