from .bertscore_evaluator import BertScoreEvaluator
from .bleu_evaluator import BleuEvaluator
from .chrf_evaluator import ChrfEvaluator
from .cider_evaluator import CiderEvaluator
from .distinct_evaluator import DistinctEvaluator
from .meteor_evaluator import MeteorEvaluator
from .nist_evaluator import NistEvaluator
from .qa_evaluator import QaEvaluator
from .rouge_evaluator import RougeEvaluator
from .selfbleu_evaluator import SelfBleuEvaluator
from .spice_evaluator import SpiceEvaluator
from .ter_evaluator import TerEvaluator
from .unique_evaluator import UniqueEvaluator


from typing import Set


evaluator_list = {
    'bert_score', 'bleu', 'chrf', 'chrf+', 'chrf++', 'cider', 'distinct', 'meteor', 'nist', 'qa', 'rouge', 'self_bleu', 'spice', 'ter', 'unique',
}


class BaseEvaluator():

    def __init__(self, config, metrics):
        self.config = config
        self.lower = config['lower_evaluation']
        self.metrics = metrics

    def evaluate(self, generate_corpus, reference_corpus, avg=False):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus
            avg: whether or not to average the metric results
        
        Returns:
            dict: such as ``{'bleu-1': xxx, 'bleu-2': yyy}``
        """
        for i, refs in enumerate(reference_corpus):
            if isinstance(refs, str):
                reference_corpus[i] = [refs]
        if self.lower:
            generate_corpus = [gen.lower() for gen in generate_corpus]
            reference_corpus = [[ref.lower() for ref in refs] for refs in reference_corpus]
        
        result_dict = {}
        for metric in self.metrics:
            if metric == 'bert_score':
                evaluator = BertScoreEvaluator(self.config)
            elif metric == 'bleu':
                evaluator = BleuEvaluator(self.config)
            elif metric in ['chrf', 'chrf+', 'chrf++']:
                evaluator = ChrfEvaluator(self.config, metric)
            elif metric == 'cider':
                evaluator = CiderEvaluator(self.config)
            elif metric == 'distinct':
                evaluator = DistinctEvaluator(self.config)
            elif metric == 'meteor':
                evaluator = MeteorEvaluator(self.config)
            elif metric == 'nist':
                evaluator = NistEvaluator(self.config)
            elif metric == 'qa':
                evaluator = QaEvaluator(self.config)
            elif metric == 'rouge':
                evaluator = RougeEvaluator(self.config)
            elif metric == 'self_bleu':
                evaluator = SelfBleuEvaluator(self.config)
            elif metric == 'spice':
                evaluator = SpiceEvaluator(self.config)
            elif metric == 'ter':
                evaluator = TerEvaluator(self.config)
            elif metric == 'unique':
                evaluator = UniqueEvaluator(self.config)
            
            metric_result = evaluator.evaluate(generate_corpus.copy(), reference_corpus.copy(), avg=avg)
            result_dict.update(metric_result)
        return result_dict
