from .abstract_evaluator import AbstractEvaluator
from sacrebleu import sentence_ter


class TerEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        super(TerEvaluator, self).__init__(config)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {'ter': []}

        for gen, ref in zip(generate_corpus, reference_corpus):
            score = sentence_ter(gen, ref).score
            results['ter'].append(score)
        return results
