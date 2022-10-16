import os
import tempfile
import logging
import numpy as np
from .abstract_evaluator import AbstractEvaluator


class RougeEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        self.rouge_type = config['rouge_type']
        self.rouge_max_ngrams = config['rouge_max_ngrams']
        self.multiref_strategy = config['multiref_strategy']
        super(RougeEvaluator, self).__init__(config)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {}
        if self.rouge_type == 'files2rouge':
            from files2rouge import settings
            from files2rouge import utils
            from pyrouge import Rouge155

            with tempfile.TemporaryDirectory() as tmpdir:
                gen_file = f"{tmpdir}/gen.txt"
                ref_file = f"{tmpdir}/ref.txt"
                with open(gen_file, 'w') as f:
                    f.write('\n'.join(generate_corpus.tokenized_text))
                with open(ref_file, 'w') as f:
                    for refs in reference_corpus.tokenized_text:
                        assert len(refs) == 1, "`files2rouge` only supports single reference."
                        f.write(refs[0] + '\n')
                sys_root, model_root = [os.path.join(tmpdir, _) for _ in ["system", "model"]]
                os.mkdir(sys_root)
                os.mkdir(model_root)

                s = settings.Settings()
                s._load()

                utils.split_files(ref_file, gen_file, model_root, sys_root)
                r = Rouge155(rouge_dir=os.path.dirname(s.data['ROUGE_path']), log_level=logging.ERROR, stemming=True)
                r.system_dir = sys_root
                r.model_dir = model_root
                r.system_filename_pattern = r's.(\d+).txt'
                r.model_filename_pattern = 'm.[A-Z].#ID#.txt'
                rouge_args_str = f"-e {s.data['ROUGE_data']} -c 95 -r 1000 -n {self.rouge_max_ngrams} -w 1.2 -a"
                output = r.convert_and_evaluate(rouge_args=rouge_args_str)

            for l in output.split('\n'):
                if l.find('Average_F') >= 0:
                    l = l.split(' ')
                    results[l[1].lower()] = float(l[3]) * 100

        elif self.rouge_type == 'rouge':
            from rouge import Rouge

            reference_corpus = [ref[0] for ref in reference_corpus.tokenized_text]
            metrics = [f'rouge-{i}' for i in range(1, self.rouge_max_ngrams + 1)] + ['rouge-l']
            rouge = Rouge(metrics=metrics)
            scores = rouge.get_scores(generate_corpus.tokenized_text, reference_corpus, avg=True)
            for i in range(1, self.rouge_max_ngrams + 1):
                results[f'rouge-{i}'] = scores[f'rouge-{i}']['f'] * 100
            results['rouge-l'] = scores['rouge-l']['f'] * 100

        elif self.rouge_type == 'py-rouge':
            from rouge import Rouge

            rouge = Rouge(
                metrics=['rouge-n', 'rouge-l'],
                max_n=self.rouge_max_ngrams,
                limit_length=False,
                apply_avg=False,
                apply_best=True,
                weight_factor=1.2
            )
            scores = rouge.get_scores(generate_corpus.text, reference_corpus.text)
            for i in range(1, self.rouge_max_ngrams + 1):
                results[f'rouge-{i}'] = scores[f'rouge-{i}']['f'] * 100
            results['rouge-l'] = scores['rouge-l']['f'] * 100

        elif self.rouge_type == 'rouge-score':
            from rouge_score import rouge_scorer

            rouge_types = [f'rouge{i}' for i in range(1, self.rouge_max_ngrams + 1)] + ["rougeLsum"]
            rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types, split_summaries=True, use_stemmer=True)
            results = {k: [] for k in [f'rouge-{i}' for i in range(1, self.rouge_max_ngrams + 1)] + ['rouge-l']}
            for gen, refs in zip(generate_corpus.text, reference_corpus.text):
                scores = [rouge.score(ref, gen) for ref in refs]
                if len(scores) > 1 and self.multiref_strategy == 'leave_one_out':
                    func = lambda x: (max(x) * (len(x) - 1) + np.partition(x, -2)[-2]) / len(x)
                else:
                    func = max

                for i in range(1, self.rouge_max_ngrams + 1):
                    results[f'rouge-{i}'].append(func([s[f'rouge{i}'].fmeasure for s in scores]) * 100)
                results['rouge-l'].append(func([s['rougeLsum'].fmeasure for s in scores]) * 100)

        elif self.rouge_type == 'pycocoevalcap':
            from pycocoevalcap.rouge.rouge import Rouge

            refs = {idx: r for idx, r in enumerate(reference_corpus.tokenized_text)}
            gen = {idx: [g] for idx, g in enumerate(generate_corpus.tokenized_text)}
            score = Rouge().compute_score(refs, gen)[0]
            results['rouge-l'] = score * 100
        return results
