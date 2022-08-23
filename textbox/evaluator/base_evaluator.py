from nltk.tokenize import word_tokenize

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


evaluator_list = {
    'bert_score', 'bleu', 'chrf', 'chrf+', 'chrf++', 'cider', 'distinct', 'meteor', 'nist', 'qa', 'rouge', 'self_bleu', 'spice', 'ter', 'unique',
}


class Text:
    def __init__(self, text, lower, tokenizer=None):
        self.text = text.lower() if lower else text
        if tokenizer:
            self.text = tokenizer.decode(tokenizer.encode(self.text), skip_special_tokens=True)
        self.tokens = word_tokenize(text)
        self.tokenized_text = ' '.join(self.tokens)
    

class Corpus:
    def __init__(self, corpus, lower, mode, tokenizer=None):
        self.mode = mode
        if mode == 'gen':
            self.corpus = [Text(text, lower, tokenizer) for text in corpus]
        else:
            self.corpus = [[Text(text, lower, tokenizer) for text in texts] for texts in corpus]

    @property
    def text(self):
        if self.mode == 'gen':
            return [text.text for text in self.corpus]
        else:
            return [[text.text for text in texts] for texts in self.corpus]
    
    @property
    def tokenized_text(self):
        if self.mode == 'gen':
            return [text.tokenized_text for text in self.corpus]
        else:
            return [[text.tokenized_text for text in texts] for texts in self.corpus]
    
    @property
    def tokens(self):
        if self.mode == 'gen':
            return [text.tokens for text in self.corpus]
        else:
            return [[text.tokens for text in texts] for texts in self.corpus]


class BaseEvaluator():

    def __init__(self, config, metrics):
        self.config = config
        self.lower = config['lower_evaluation']
        self.metrics = metrics

    def _process_corpus(self, generate_corpus, reference_dataset):
        reference_corpus = reference_dataset.target_text
        for i, refs in enumerate(reference_corpus):
            if isinstance(refs, str):
                reference_corpus[i] = [refs]
        tokenizer = reference_dataset.tokenizer if self.config['is_chinese_task'] else None
        generate_corpus = Corpus(generate_corpus, self.lower, 'gen', tokenizer)
        reference_corpus = Corpus(reference_corpus, self.lower, 'ref', tokenizer)
        return generate_corpus, reference_corpus

    def evaluate(self, generate_corpus, reference_dataset, avg=True):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_dataset: the referenced dataset
            avg: whether or not to average the metric results
        
        Returns:
            dict: such as ``{'bleu-1': xxx, 'bleu-2': yyy}``
        """
        generate_corpus, reference_corpus = self._process_corpus(generate_corpus, reference_dataset)
        
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
            
            metric_result = evaluator.evaluate(generate_corpus, reference_corpus, avg=avg)
            result_dict.update(metric_result)
        return result_dict
