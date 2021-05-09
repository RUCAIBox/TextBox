# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE
# @Time   : 2021/4/12
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.abstract_evaluator
#####################################
"""

import numpy as np


class AbstractEvaluator(object):
    """:class:`AbstractEvaluator` is an abstract object which supports
    the evaluation of the model. It is called by :class:`Trainer`.

    Note:       
        If you want to inherit this class and implement your own evalautor class, 
        you must implement the following functions.

    Args:
        config (Config): The config of evaluator.

    """

    def evaluate(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus
        
        Returns:
            dict: such as ``{metric-1: xxx}``
        """
        # get metrics
        metric_dict = {}
        info_dict = self._calc_metrics_info(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
        for key in info_dict:
            tp_list = info_dict[key]
            tp_val = np.mean(tp_list)
            metric_dict[key] = round(tp_val, 4)
        return metric_dict

    def _calc_metrics_info(self):
        """ to calculate the metrics"""
        raise NotImplementedError
