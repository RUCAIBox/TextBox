# @Time   : 2020/11/14
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

# UPDATE
# @Time   : 2021/4/12
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.base_evaluator
#######################################
"""

from textbox.evaluator.bleu_evaluator import *
from textbox.evaluator.distinct_evaluator import *
from textbox.evaluator.rouge_evaluator import *
from textbox.evaluator.selfbleu_evaluator import *

class BaseEvaluator():
    def __init__(self, config):
        self.config = config
        self.metrics = config["metrics"]
        self._check_args()
        # [1, 2, 3, 4]
    
    def evaluate(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus
        
        Returns:
            dict: such as ``{'bleu-1': xxx, 'bleu-1-avg': yyy}``
        """
        result_dict = {}
        for metric in self.metrics:
            if metric == 'bleu':
                if self.config['task_type'].lower() == "unconditional":
                    per_gen_ref = False
                else:
                    per_gen_ref = True
                evaluator = BleuEvaluator(per_gen_ref)
            elif metric == 'self_bleu':
                evaluator = SelfBleuEvaluator()
            elif metric == 'rouge':
                self.max_target_length = self.config["max_target_length"]
                evaluator = RougeEvaluator(self.max_target_length)
            elif metric == 'distinct':
                evaluator = DistinctEvaluator()
            metric_result = evaluator.evaluate(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
            result_dict[metric] = metric_result
        return result_dict

    def _check_args(self):
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                if self.metrics[0] == '[':
                    self.metrics = self.metrics[1: ]
                if self.metrics[-1] == ']':
                    self.metrics = self.metrics[: len(self.metrics) - 1]
                self.metrics = self.metrics.strip().split(",")
            self.metrics = [metric.lower() for metric in self.metrics]
            for metric in self.metrics:
                if metric not in ['bleu', 'self_bleu', 'rouge', 'distinct']:
                    raise ValueError("evaluator {} can't be found. (evaluator should be in [\"bleu\", \"self-bleu\", \"rouge\", \"distinct\"]).".format(metric))
        else:
            raise TypeError('evaluator must be a string or list')
        

