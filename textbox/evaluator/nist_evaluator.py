from .utils.pymteval import NISTScore
from .abstract_evaluator import AbstractEvaluator


class NistEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        super(NistEvaluator, self).__init__(config)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {}

        nist = NISTScore()
        for gen, refs in zip(generate_corpus.text, reference_corpus.text):
            nist.append(gen, refs)
        results['nist'] = nist.score()
        return results
