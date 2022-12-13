from .abstract_evaluator import AbstractEvaluator
from pycocoevalcap.cider.cider import Cider


class CiderEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        super(CiderEvaluator, self).__init__(config)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {}
        refs = {idx: r for idx, r in enumerate(reference_corpus.tokenized_text)}
        gen = {idx: [g] for idx, g in enumerate(generate_corpus.tokenized_text)}
        score = Cider().compute_score(refs, gen)[0]
        results['CIDEr'] = score * 10
        return results
