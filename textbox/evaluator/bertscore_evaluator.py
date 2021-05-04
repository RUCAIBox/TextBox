# @Time   : 2021/5/1
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.bertscore_evaluator
#######################################
"""

import logging
import transformers
import numpy as np
from bert_score import score
from textbox.evaluator.abstract_evaluator import AbstractEvaluator

class BertScoreEvaluator(AbstractEvaluator):
    r"""Bert Score Evaluator. Now, we support metrics `'bert score'`.
    """
    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers
        self.lang = "en"
        
    def _preprocess(self, input_sentence):
        return " ".join(input_sentence)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):

        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)

        generate_corpus = [self._preprocess(generate_sentence) for generate_sentence in generate_corpus]
        reference_corpus = [self._preprocess(reference_sentence) for reference_sentence in reference_corpus]
        
        result = {}
        if self.model == None:
            P, R, F1 = score(generate_corpus, reference_corpus, lang=self.lang, verbose=False)
        else:
            if self.num_layers == None:
                raise ValueError("num_layer should be an integer")
            P, R, F1 = score(generate_corpus, reference_corpus, model_type=self.model, num_layers=self.num_layers, verbose=False)
        result['bert-score'] = F1.tolist()
        return result