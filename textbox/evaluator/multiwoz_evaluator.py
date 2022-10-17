import json
from copy import deepcopy
from .abstract_evaluator import AbstractEvaluator
from .utils.multiwoz.config import Config
from .utils.multiwoz.reader import MultiWozReader
from .utils.multiwoz.eval import MultiWozEvaluator


class MultiwozEvaluator(AbstractEvaluator):
    r"""Multiwoz Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        super(MultiwozEvaluator, self).__init__(config)
        self.cfg = Config()
        self.reader = MultiWozReader(self.cfg)
        self.eval = MultiWozEvaluator(self.reader, self.cfg)

    def load_data(self, group):
        with open(f'dataset/multiwoz/{group}.json', 'r') as f:
            self.data = json.load(f)
        self.turn_domains = [utt['turn_domain'] for utt in self.data]

    def span_db(self, bspan, turn_domain):
        return self.reader.bspan_to_DBpointer(bspan, turn_domain)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        i = 0
        data = deepcopy(self.data)
        for utt in data:
            utt['bspn_gen'] = generate_corpus.text[i * 3]
            utt['aspn_gen'] = generate_corpus.text[i * 3 + 1]
            utt['resp_gen'] = generate_corpus.text[i * 3 + 2]
            i += 1
        bleu, success, inform = self.eval.validation_metric(data)
        results = {'bleu': bleu, 'success': success, 'inform': inform, 'overall': bleu + (success + inform) / 2}
        return results
