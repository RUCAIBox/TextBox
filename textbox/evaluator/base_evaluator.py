from textbox.evaluator.bleu_evaluator import BleuEvaluator
from textbox.evaluator.distinct_evaluator import DistinctEvaluator
from textbox.evaluator.selfbleu_evaluator import SelfBleuEvaluator
from textbox.evaluator.averagelength_evaluator import AvgLenEvaluator
from textbox.evaluator.cider_evaluator import CIDErEvaluator
from textbox.evaluator.chrfplusplus_evaluator import ChrfPlusPlusEvaluator
from textbox.evaluator.meteor_evaluator import MeteorEvaluator
from textbox.evaluator.bertscore_evaluator import BertScoreEvaluator
from textbox.evaluator.unique_evaluator import UniqueEvaluator
from textbox.evaluator.rouge_evaluator import RougeEvaluator

from typing import Iterable

evaluator_list = {
    'bleu', 'self_bleu', 'rouge', 'distinct', 'nll_test', 'avg_len', 'cider', 'chrf++', 'meteor', 'unique',
    'bert_score'
}


class BaseEvaluator:

    def __init__(self, config, metrics: Iterable[str]):
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
            evaluator = None
            if metric == 'bleu':
                task_type = (self.config['task_type'] == "unconditional")
                evaluator = BleuEvaluator(task_type)
            elif metric == 'self_bleu':
                if self.config['task_type'] == "unconditional":
                    evaluator = SelfBleuEvaluator()
                else:
                    raise ValueError("task_type should be 'unconditional' for self-bleu")
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
                num_layers = self.config['num_layers']
                evaluator = BertScoreEvaluator(model, num_layers)
            elif metric == 'unique':
                evaluator = UniqueEvaluator()
            elif metric == 'nll_test':
                continue
            if evaluator is None:
                return dict()
            metric_result = evaluator.evaluate(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
            result_dict[metric] = metric_result
        return result_dict
