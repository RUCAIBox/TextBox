from .abstract_evaluator import AbstractEvaluator


class ChrfEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config, metric):
        super(ChrfEvaluator, self).__init__(config)
        self.metric = metric
        self.chrf_type = config['chrf_type']

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {}
        word_order = self.metric.count('+')
        if self.chrf_type == 'm-popovic':
            from .utils.chrf import computeChrF

            score = computeChrF(reference_corpus.text, generate_corpus.text, nworder=word_order)
            results[self.metric] = score * 100

        elif self.chrf_type == 'sacrebleu':
            from sacrebleu import corpus_chrf

            score = corpus_chrf(generate_corpus.text, reference_corpus.text, word_order=word_order)
            results[self.metric] = score.score
        return results
