import logging
import transformers
from bert_score import score
from .abstract_evaluator import AbstractEvaluator


class BertScoreEvaluator(AbstractEvaluator):
    r"""Bert Score Evaluator. Now, we support metrics `'bert score'`.
    """

    def __init__(self, config):
        super(BertScoreEvaluator, self).__init__(config)
        self.model_type = config['bert_score_model_type']
        self.lang = config['tgt_lang']
        self.batch_size = config['eval_batch_size']
        self.device = config['device']

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)

        results = {}
        _, _, f_score = score(
            generate_corpus.text, reference_corpus.text, lang=self.lang, batch_size=self.batch_size, device=self.device
        )
        results['bertscore'] = f_score.mean().item() * 100
        return results
