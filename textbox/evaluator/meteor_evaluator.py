from .abstract_evaluator import AbstractEvaluator


class MeteorEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        super(MeteorEvaluator, self).__init__(config)
        self.meteor_type = config['meteor_type']
        self.corpus_meteor = config['corpus_meteor']

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {}
        if self.meteor_type == 'pycocoevalcap':
            from pycocoevalcap.meteor.meteor import Meteor

            refs = {idx: r for idx, r in enumerate(reference_corpus.tokenized_text)}
            gen = {idx: [g] for idx, g in enumerate(generate_corpus.tokenized_text)}
            score = Meteor().compute_score(refs, gen)
            if self.corpus_meteor:
                score = score[0]
            else:
                score = [s * 100 for s in score[1]]
            results['meteor'] = score * 100
        else:
            from nltk.translate.meteor_score import meteor_score

            results['meteor'] = []
            for gen, refs in zip(generate_corpus.tokenized_text, reference_corpus.tokenized_text):
                score = meteor_score(refs, gen)
                results['meteor'].append(score * 100)
        return results
