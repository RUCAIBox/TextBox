# @Time   : 2021/4/19
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.meteor_evaluator
#######################################
"""

import numpy as np
from nltk.translate.meteor_score import meteor_score
from textbox.evaluator.abstract_evaluator import AbstractEvaluator


class MeteorEvaluator(AbstractEvaluator):

    def _preprocess(self, input_sentence):
        return " ".join(input_sentence)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        generate_corpus = [self._preprocess(generate_sentence) for generate_sentence in generate_corpus]
        reference_corpus = [self._preprocess(reference_sentence) for reference_sentence in reference_corpus]
        reference_corpus = [[reference_sentence] for reference_sentence in reference_corpus]

        result = {}
        scores = []
        for gen, refs in zip(generate_corpus, reference_corpus):
            score = meteor_score(refs, gen)
            scores.append(score)

        result['meteor'] = scores
        return result
