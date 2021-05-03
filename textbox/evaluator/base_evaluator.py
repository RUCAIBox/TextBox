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
from textbox.evaluator.averagelength_evaluator import *
from textbox.evaluator.cider_evaluator import *
from textbox.evaluator.chrfplusplus_evaluator import *
from textbox.evaluator.meteor_evaluator import *
from textbox.evaluator.bertscore_evaluator import *
from textbox.evaluator.unique_evaluator import *

evaluator_list = ['bleu', 'self_bleu', 'rouge', 'distinct', 'nll_test', 'avg_len', 'cider', 'chrf++', 'meteor', 'unique', 'bert_score']

class BaseEvaluator():
    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics
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
                per_gen_ref = (self.config['task_type'].lower() == "unconditional")
                evaluator = BleuEvaluator(per_gen_ref)
            elif metric == 'self_bleu':
                evaluator = SelfBleuEvaluator()
            elif metric == 'rouge':
                evaluator = RougeEvaluator()
            elif metric == 'distinct':
                evaluator = DistinctEvaluator()
            elif metric == 'avg_len':
                evaluator = AvgLenEvaluator()
            elif metric == 'cider':
                evaluator = CIDErEvaluator()
            elif metric == 'chrf++':
                evaluator = ChrfPlusPlusEvaluator()
            elif metric == 'meteor':
                evaluator = MeteorEvaluator()
            elif metric == 'bert_score':
                model = self.config['bert_score_model_path']
                layers = self.config['num_layers']
                evaluator = BertScoreEvaluator(model, layers)
            elif metric == 'unique':
                evaluator = UniqueEvaluator()
            elif metric == 'nll_test':
                continue
            metric_result = evaluator.evaluate(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
            result_dict[metric] = metric_result
        return result_dict
        

