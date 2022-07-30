import numpy as np


class AbstractEvaluator():
    """:class:`AbstractEvaluator` is an abstract object which supports
    the evaluation of the model. It is called by :class:`Trainer`.

    Note:       
        If you want to inherit this class and implement your own evalautor class, 
        you must implement the following functions.

    Args:
        config (Config): The config of evaluator.

    """

    def __init__(self, config):
        self.config = config

    def evaluate(self, generate_corpus, reference_corpus, avg=True):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus
            avg: whether or not to average the metric results
        
        Returns:
            dict: such as ``{metric-1: xxx}``
        """
        metric_dict = self._calc_metrics_info(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
        if avg:
            for k, v in metric_dict.items():
                if isinstance(v, list) or isinstance(v, float):
                    metric_dict[k] = round(np.mean(v), 2 if k != 'cider' else 3)
        return metric_dict

    def _calc_metrics_info(self):
        """ to calculate the metrics"""
        raise NotImplementedError
