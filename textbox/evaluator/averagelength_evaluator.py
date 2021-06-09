# @Time   : 2021/4/19
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.averagelength_evaluator
##########################################
"""

import numpy as np
from textbox.evaluator.abstract_evaluator import AbstractEvaluator


class AvgLenEvaluator(AbstractEvaluator):

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        result = {}
        length = []
        for sentence in generate_corpus:
            length.append(len(sentence))
        result['avg-length'] = length
        return result
