import os
import subprocess
import nltk
import tempfile
import warnings
import traceback
from itertools import zip_longest
from packaging import version
from .abstract_evaluator import AbstractEvaluator


class BleuEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, config):
        super(BleuEvaluator, self).__init__(config)
        self.bleu_type = config['bleu_type']
        self.max_ngrams = config['bleu_max_ngrams']
        self.ngrams = ['bleu-{}'.format(n) for n in range(1, self.max_ngrams + 1)]
        self.smoothing_function = config['smoothing_function']
        self.corpus_bleu = config['corpus_bleu']
        self.sacrebleu_romanian = self.config['sacrebleu_romanian']
        if self.bleu_type == 'nltk' and self.smoothing_function > 0 and config['dataset'] in ['pc', 'dd']:
            nltk_version = version.parse(nltk.__version__)
            if nltk_version != version.parse('3.5'):
                warnings.warn("The version of `NLTK` should be 3.5 to reproduce results.")
        self._generate_weights()

    def _generate_weights(self):
        self.ngram_weights = []
        for n in range(1, self.max_ngrams + 1):
            weights = [0.] * self.max_ngrams
            weights[:n] = [1. / n] * n
            self.ngram_weights.append(weights)

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {}
        if self.bleu_type in ['nltk', 'fast-bleu', 'pycocoevalcap']:
            for ngram in self.ngrams:
                results[ngram] = []

        if self.bleu_type == 'fast-bleu':
            from fast_bleu import bleu

            for i, refs in enumerate(reference_corpus.tokens):
                assert len(refs) == 1, "`fast-bleu` only supports single reference."
                reference_corpus[i] = refs[0]
            bleu = bleu(reference_corpus.tokens, dict(zip(self.ngrams, self.ngram_weights)))
            scores = bleu.get_score(generate_corpus.tokens)
            for ngram in self.ngrams:
                results[ngram] = [s * 100 for s in scores[ngram]]

        elif self.bleu_type == 'nltk':
            from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

            if self.corpus_bleu:
                for ngram, weights in zip(self.ngrams, self.ngram_weights):
                    score = corpus_bleu(
                        reference_corpus.tokens, generate_corpus.tokens, weights,
                        getattr(SmoothingFunction(), f"method{self.smoothing_function}")
                    )
                    results[ngram] = score * 100
            else:  # sentence_bleu
                for gen, refs in zip(generate_corpus.tokens, reference_corpus.tokens):
                    for ngram, weights in zip(self.ngrams, self.ngram_weights):
                        score = sentence_bleu(
                            refs, gen, weights, getattr(SmoothingFunction(), f"method{self.smoothing_function}")
                        )
                        results[ngram].append(score * 100)

        elif self.bleu_type == 'mt-eval':
            from .utils.pymteval import BLEUScore

            bleu = BLEUScore()
            for gen, refs in zip(generate_corpus.text, reference_corpus.text):
                bleu.append(gen, refs)
            results['bleu'] = bleu.score() * 100

        elif self.bleu_type == 'sacrebleu':
            import sacrebleu

            reference_corpus = list(zip_longest(*reference_corpus.text))
            bleu = sacrebleu.corpus_bleu(generate_corpus.text, reference_corpus)
            results['bleu'] = bleu.score
            results['bleu-precisions'] = bleu.prec_str

        elif self.bleu_type == 'pycocoevalcap':
            from pycocoevalcap.bleu.bleu import Bleu
            refs = {idx: r for idx, r in enumerate(reference_corpus.tokenized_text)}
            gen = {idx: [g] for idx, g in enumerate(generate_corpus.tokenized_text)}
            scores = Bleu(4).compute_score(refs, gen, verbose=0)[0]
            for ngram, score in zip(self.ngrams, scores):
                results[ngram] = score * 100

        elif self.bleu_type == 'multi-bleu':
            reference_corpus = list(zip_longest(*reference_corpus.tokenized_text))
            max_ref_num = len(reference_corpus)

            with tempfile.TemporaryDirectory() as tmpdir:
                gen_file = f"{tmpdir}/gen.txt"
                ref_file = f"{tmpdir}/ref.txt"
                with open(f"{gen_file}", "w") as f:
                    f.write("\n".join(generate_corpus.tokenized_text))
                for i in range(max_ref_num):
                    with open(f"{ref_file}{i}", "w") as f:
                        f.write('\n'.join([ref or '' for ref in reference_corpus[i]]))
                try:
                    ref_file = f"{ref_file}[{','.join([str(i) for i in range(max_ref_num)])}]"
                    scores = subprocess.check_output(
                        f"perl textbox/evaluator/utils/multi-bleu.perl {ref_file} < {gen_file}",
                        stderr=subprocess.STDOUT,
                        shell=True
                    )
                    scores = scores.decode().strip().split('\n')[-1].split()
                    results['bleu'] = float(scores[2][:-1])
                    results['bleu-precisions'] = scores[3]
                except subprocess.CalledProcessError as call_e:
                    traceback.print_exc()
                    print(call_e.output.decode().strip())
                    exit(0)

        elif self.bleu_type == 'sacrebleu-romanian':
            with tempfile.TemporaryDirectory() as tmpdir:
                gen_file = f"{tmpdir}/gen.txt"
                ref_file = f"{tmpdir}/ref.txt"
                tmp_file = f"{tmpdir}/tmp.txt"
                with open(f"{gen_file}", "w") as f:
                    f.write("\n".join(generate_corpus.tokenized_text))
                with open(f"{ref_file}", "w") as f:
                    for refs in reference_corpus.tokenized_text:
                        assert len(refs) == 1, "`sacrebleu-romanian` only supports single reference."
                        f.write(refs[0] + '\n')
                os.system(f'bash {self.sacrebleu_romanian} {gen_file} {ref_file} > {tmp_file}')
                with open(tmp_file, 'r') as fin:
                    results['bleu'] = float(fin.readline().strip())

        return results
