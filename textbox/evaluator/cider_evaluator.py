# @Time   : 2021/4/19
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.cider_evaluator
#######################################
"""

import numpy as np
from nltk import stem
from collections import defaultdict
from textbox.evaluator.abstract_evaluator import AbstractEvaluator

class CIDErEvaluator(AbstractEvaluator):
    r"""CIDEr Evaluator. Now, we support metrics `'CIDEr'`.
    """

    def __init__(self):
        self.n_grams = [1, 2, 3, 4]
        self.stem_generator = stem.SnowballStemmer("english")

        self.total_num = 0
        self.generate_corpus = []
        self.reference_corpus = []
        self.gen_corpus_count = []
        self.ref_corpus_count = []
        self.gen_document_frequency = defaultdict(int)
        self.ref_document_frequency = defaultdict(int)

    def _get_stem(self, input_sentence):
        res = []
        for word in input_sentence:
            res.append(self.stem_generator.stem(word))
        return res
    
    def _generate_ngrams(self, input_sentence):
        r"""
        """
        ngrams_counts = defaultdict(int)
        for n_gram in range(1, max(self.n_grams) + 1):
            for index in range(len(input_sentence) - n_gram + 1):
                cur_ngram = tuple(input_sentence[index: index + n_gram])
                ngrams_counts[cur_ngram] += 1
        return ngrams_counts
    
    def _generate_ngrams_count(self):
        r"""
        """
        for i in range(self.total_num):
            gen_result = self._generate_ngrams(self.generate_corpus[i])
            ref_result = [self._generate_ngrams(reference_sentence) for reference_sentence in self.reference_corpus[i]]
            self.gen_corpus_count.append(gen_result)
            self.ref_corpus_count.append(ref_result)
    
    def _count_document_times(self):
        r"""
        """
        for gen in self.gen_corpus_count:
            for ngram in set([ngram for (ngram, count) in gen.items()]):
                self.gen_document_frequency[ngram] += 1
        for refs in self.ref_corpus_count:
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.ref_document_frequency[ngram] += 1
    
    def _generate_vector(self, ngram_count, corpus_type):
        r"""
        Args:
            ngram_count (defaultdict[List[str]: int])
            corpus_type (str)
        """
        if corpus_type.lower() == "generate":
            document_frequency = self.gen_document_frequency
        elif corpus_type.lower() == "reference":
            document_frequency = self.ref_document_frequency
        else:
            raise ValueError("Corpus type should be in ['generate', 'reference']")

        vec = [defaultdict(float) for _ in self.n_grams]
        norm = [0. for _ in self.n_grams]
        
        for (ngram, times) in ngram_count.items():
            tf = times / sum(ngram_count.values())
            df = max(1.0, document_frequency[ngram])
            if self.total_num == 1:
                idf = 1
            else:
                idf = np.log(self.total_num) - np.log(df)
            index = self.n_grams.index(len(ngram))
            vec[index][ngram] = tf * idf
            norm[index] += pow(vec[index][ngram], 2)
        
        norm = [np.sqrt(val) for val in norm]
        return vec, norm

    def _cal_cosine_similarity(self, gen_vec, gen_norm, ref_vec, ref_norm):
        val = np.array([0. for _ in self.n_grams])
        for index in range(len(self.n_grams)):
            numerator = 0
            for ngram in gen_vec[index].keys():
                numerator += gen_vec[index][ngram] * ref_vec[index][ngram]
            val[index] = numerator / (gen_norm[index] * ref_norm[index])
        return val

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus(List[List[str]]): the generated corpus
            reference_corpus(List[List[str]]): the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.n_grams
        """
        self.total_num = len(generate_corpus)
        reference_corpus = [[reference_sentence] for reference_sentence in reference_corpus]

        for i in range(self.total_num):
            self.generate_corpus.append(self._get_stem(generate_corpus[i]))
            self.reference_corpus.append([self._get_stem(reference_sentence) for reference_sentence in reference_corpus[i]])
        
        self._generate_ngrams_count()
        self._count_document_times()

        scores = []
        for (gen_ngram, ref_ngrams) in zip(self.gen_corpus_count, self.ref_corpus_count):
            cider_n_score = np.array([0. for _ in self.n_grams])
            gen_vec, gen_norm = self._generate_vector(gen_ngram, corpus_type="generate")
            for ref_ngram in ref_ngrams:
                ref_vec, ref_norm = self._generate_vector(ref_ngram, corpus_type="reference")
                cosine_score = self._cal_cosine_similarity(gen_vec, gen_norm, ref_vec, ref_norm)
                cider_n_score += cosine_score
            cider_n_score /= len(ref_ngrams)
            scores.append(cider_n_score)
        scores = np.array(scores)

        result = {}
        result['CIDEr'] = np.mean(scores, axis=0)
        return result
    
    def __str__(self):
        mesg = 'The CIDEr Evaluator Info:\n' + '\tMetrics:[CIDEr], Ngram:[' + ', '.join(map(str, self.n_grams)) + ']'
        return mesg