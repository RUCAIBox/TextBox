from collections import Counter
from .abstract_evaluator import AbstractEvaluator


class DistinctEvaluator(AbstractEvaluator):
    r"""Distinct Evaluator. Now, we support metrics `'distinct'`.
    """

    def __init__(self, config):
        super(DistinctEvaluator, self).__init__(config)
        self.distinct_max_ngrams = config['distinct_max_ngrams']
        self.inter_distinct = config['inter_distinct']

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        results = {}

        if self.inter_distinct:
            ngrams_all = [Counter() for _ in range(self.distinct_max_ngrams)]
        else:
            scores = [[] for _ in range(self.distinct_max_ngrams)]

        for gen in generate_corpus.tokens:
            ngrams = []
            for i in range(self.distinct_max_ngrams):
                ngrams.append(gen[i:])
                ngram = Counter(zip(*ngrams))
                if self.inter_distinct:
                    ngrams_all[i].update(ngram)
                else:
                    scores[i].append((len(ngram) + 1e-12) / (max(0, len(gen) - i) + 1e-5) * 100)

        for i in range(self.distinct_max_ngrams):
            if self.inter_distinct:
                results[f'distinct-{i+1}'] = (len(ngrams_all[i]) +
                                              1e-12) / (sum(ngrams_all[i].values()) +
                                                        1e-5) * 100 if self.inter_distinct else scores[i]

        return results
