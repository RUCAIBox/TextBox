from nltk.tokenize import word_tokenize

evaluator_list = {
    'bert_score',
    'bleu',
    'chrf',
    'chrf+',
    'chrf++',
    'cider',
    'distinct',
    'meteor',
    'multiwoz',
    'nist',
    'qa',
    'rouge',
    'self_bleu',
    'spice',
    'style',
    'ter',
    'unique',
}

PUNCTUATIONS = [
    "''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!", ",", ":", "-", "--", "...", ";"
]


class Text:

    def __init__(self, text, lower, tokenizer=None, remove_punc=False):
        self.text = text.lower() if lower else text
        if tokenizer:
            self.text = tokenizer.decode(tokenizer.encode(self.text), skip_special_tokens=True)
        self.text = self.text or 'UNK'
        self.tokens = word_tokenize(self.text)
        if remove_punc:
            self.tokenized_text = ' '.join([token for token in self.tokens if token not in PUNCTUATIONS])
        else:
            self.tokenized_text = ' '.join(self.tokens)


class Corpus:

    def __init__(self, corpus, lower, mode, tokenizer=None, remove_punc=False):
        self.mode = mode
        if mode == 'gen':
            self.corpus = [Text(text, lower, tokenizer, remove_punc) for text in corpus]
        else:
            self.corpus = [[Text(text, lower, tokenizer, remove_punc) for text in texts] for texts in corpus]

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
        self.remove_punc = config['remove_punc']
        self.metrics = metrics

        self.evaluators = []
        for metric in self.metrics:
            if metric == 'bert_score':
                from .bertscore_evaluator import BertScoreEvaluator
                evaluator = BertScoreEvaluator(self.config)
            elif metric == 'bleu':
                from .bleu_evaluator import BleuEvaluator
                evaluator = BleuEvaluator(self.config)
            elif metric in ['chrf', 'chrf+', 'chrf++']:
                from .chrf_evaluator import ChrfEvaluator
                evaluator = ChrfEvaluator(self.config, metric)
            elif metric == 'cider':
                from .cider_evaluator import CiderEvaluator
                evaluator = CiderEvaluator(self.config)
            elif metric == 'distinct':
                from .distinct_evaluator import DistinctEvaluator
                evaluator = DistinctEvaluator(self.config)
            elif metric == 'meteor':
                from .meteor_evaluator import MeteorEvaluator
                evaluator = MeteorEvaluator(self.config)
            elif metric == 'multiwoz':
                from .multiwoz_evaluator import MultiwozEvaluator
                evaluator = MultiwozEvaluator(self.config)
            elif metric == 'nist':
                from .nist_evaluator import NistEvaluator
                evaluator = NistEvaluator(self.config)
            elif metric == 'qa':
                from .qa_evaluator import QaEvaluator
                evaluator = QaEvaluator(self.config)
            elif metric == 'rouge':
                from .rouge_evaluator import RougeEvaluator
                evaluator = RougeEvaluator(self.config)
            elif metric == 'self_bleu':
                from .selfbleu_evaluator import SelfBleuEvaluator
                evaluator = SelfBleuEvaluator(self.config)
            elif metric == 'spice':
                from .spice_evaluator import SpiceEvaluator
                evaluator = SpiceEvaluator(self.config)
            elif metric == 'style':
                from .style_evaluator import StyleEvaluator
                evaluator = StyleEvaluator(self.config)
            elif metric == 'ter':
                from .ter_evaluator import TerEvaluator
                evaluator = TerEvaluator(self.config)
            elif metric == 'unique':
                from .unique_evaluator import UniqueEvaluator
                evaluator = UniqueEvaluator(self.config)

            if metric != 'hm':
                self.evaluators.append(evaluator)

    def _process_corpus(self, generate_corpus, reference_dataset):
        reference_corpus = reference_dataset.target_text
        for i, refs in enumerate(reference_corpus):
            if isinstance(refs, str):
                reference_corpus[i] = [refs]
        tokenizer = reference_dataset.tokenizer if self.config['is_chinese_task'] else None
        generate_corpus = Corpus(generate_corpus, self.lower, 'gen', tokenizer, self.remove_punc)
        reference_corpus = Corpus(reference_corpus, self.lower, 'ref', tokenizer, self.remove_punc)
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
        for evaluator in self.evaluators:
            metric_result = evaluator.evaluate(generate_corpus, reference_corpus, avg=avg)
            result_dict.update(metric_result)
        if 'hm' in self.metrics:
            result_dict['hm'] = 2 / (1 / result_dict['bleu'] + 1 / result_dict['style'])
        return result_dict
