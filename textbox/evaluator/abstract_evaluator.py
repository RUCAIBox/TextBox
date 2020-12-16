# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

"""
textbox.evaluator.abstract_evaluator
#####################################
"""


class AbstractEvaluator(object):
    """:class:`AbstractEvaluator` is an abstract object which supports
    the evaluation of the model. It is called by :class:`Trainer`.

    Note:       
        If you want to inherit this class and implement your own evalautor class, 
        you must implement the following functions.

    Args:
        config (Config): The config of evaluator.

    """

    def __init__(self, config):
        self.metrics = config['metrics']

    def _check_args(self):
        """check the correct of the setting"""
        raise NotImplementedError

    def collect(self):
        """get the intermediate results for each batch, it is called at the end of each batch"""
        raise NotImplementedError

    def evaluate(self):
        """calculate the metrics of all batches, it is called at the end of each epoch"""
        raise NotImplementedError

    def metrics_info(self):
        """get metrics result"""
        raise NotImplementedError

    def _calculate_metrics(self):
        """ to calculate the metrics"""
        raise NotImplementedError
