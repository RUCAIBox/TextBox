import pickle
import collections
import sys

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

from textbox.evaluator.abstract_evaluator import AbstractEvaluator


class Kb2TextEvaluator(AbstractEvaluator):
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, generate_corpus, reference_corpus):
        return self._calc_metrics_info(generate_corpus, reference_corpus)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        # make dictionary
        ref = {}
        hypo = {}
        for i in range(len(generate_corpus)):
            ref_item = [" ".join(reference_corpus[i]).rstrip()]
            hypo_item = [" ".join(generate_corpus[i]).rstrip()]
            ref[i] = ref_item
            hypo[i] = hypo_item

        # compute scores
        final_scores = self.score(ref, hypo)

        return final_scores
