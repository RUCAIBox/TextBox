from fast_bleu import SelfBLEU
from .abstract_evaluator import AbstractEvaluator


class SelfBleuEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'self-bleu'`.
    """

    def __init__(self, config):
        super(SelfBleuEvaluator, self).__init__(config)
        self.max_ngrams = config['self_bleu_max_ngrams']
        self.ngrams = ['self-bleu-{}'.format(n) for n in range(1, self.max_ngrams + 1)]
        self._generate_weights()

    def _generate_weights(self):
        self.ngram_weights = []
        for n in range(1, self.max_ngrams + 1):
            weights = [0.] * self.max_ngrams
            weights[:n] = [1. / n] * n
            self.ngram_weights.append(weights)

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.n_grams
        """

        results = {}
        for ngram in self.ngrams:
            results[ngram] = []

        self_bleu = SelfBLEU(generate_corpus.tokens, dict(zip(self.ngrams, self.ngram_weights)))
        scores = self_bleu.get_score()
        for ngram in self.ngrams:
            results[ngram] = [s * 100 for s in scores[ngram]]
        return results
