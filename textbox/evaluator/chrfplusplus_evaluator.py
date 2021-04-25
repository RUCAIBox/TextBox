# @Time   : 2021/4/19
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.chrf++_evaluator
#######################################
"""
import re
import numpy as np
from nltk.util import ngrams
from collections import defaultdict, Counter
from textbox.evaluator.abstract_evaluator import AbstractEvaluator

class ChrfPlusPlusEvaluator(AbstractEvaluator):

    def __init__(self):
        self.char_n_grams = [1, 2, 3, 4, 5, 6]
        self.word_n_grams = [1, 2]
        self.beta = 3.0

    def _preprocess(self, input_sentence, ignore_whitespace=True):
        if isinstance(input_sentence, (str, list)):
            if isinstance(input_sentence, list):
                input_sentence = " ".join(input_sentence)
        else:
            raise TypeError('Input must be a string or list')

        if ignore_whitespace:
            input_sentence = re.sub("\s+", "", input_sentence)
        return input_sentence

    def _generate_ngrams(self, input_sentence, task_type):
        result = defaultdict(lambda: Counter())
        if task_type.lower() == "char":
            ngrams_list = self.char_n_grams
        elif task_type.lower() == "word":
            ngrams_list = self.word_n_grams
        else:
            raise KeyError("Task type should be in ['char', 'word']")
        for ngram in ngrams_list:
            ngram_dict = Counter(ngrams(input_sentence, ngram))
            result[ngram] = ngram_dict
        return result        

    def _ngrams_match(self, gen_ngrams, ref_ngrams):
        matchNgramCount = defaultdict(float)
        totalGenNgramCount = defaultdict(float)
        totalRefNgramCount = defaultdict(float)

        for index in ref_ngrams:
            for ngram in gen_ngrams[index]:
                totalGenNgramCount[index] += gen_ngrams[index][ngram]
            for ngram in ref_ngrams[index]:
                totalRefNgramCount[index] += ref_ngrams[index][ngram]
                if ngram in gen_ngrams[index]:
                    matchNgramCount[index] += min(gen_ngrams[index][ngram], ref_ngrams[index][ngram])
        return matchNgramCount, totalGenNgramCount, totalRefNgramCount
    
    def _calc_F(self, matchCount, genCount, refCount, beta=3.0):
        ngramF = defaultdict(float)
        ngramRecall = defaultdict(float)
        ngramPrec = defaultdict(float)

        for index in matchCount:
            if genCount[index] > 0:
                ngramPrec[index] = matchCount[index] / genCount[index]
            else:
                ngramPrec[index] = 0
            if refCount[index] > 0:
                ngramRecall[index] = matchCount[index] / refCount[index]
            else:
                ngramRecall[index] = 0
            denominator = pow(beta, 2) * ngramPrec[index] + ngramRecall[index]
            if denominator > 0:
                ngramF[index] = (1 + pow(beta, 2)) * ngramPrec[index] * ngramRecall[index] / denominator
            else:
                ngramF[index] = 0
        return ngramF, ngramRecall, ngramPrec

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus (List[List[str]]): the generated corpus
            reference_corpus (List[List[str]]): the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.n_grams
        """
        reference_corpus = [[reference_sentence] for reference_sentence in reference_corpus]

        totalMatchWordCount = defaultdict(float)
        totalRefWordCount = defaultdict(float)
        totalGenWordCount = defaultdict(float)
        totalMatchCharCount = defaultdict(float)
        totalRefCharCount = defaultdict(float)
        totalGenCharCount = defaultdict(float)
        avgTotalF = .0

        result = {}

        generate_corpus_process = [self._preprocess(generate_sentence) for generate_sentence in generate_corpus]
        reference_corpus_process = []
        for reference_sentences in reference_corpus:
            reference_corpus_process.append([self._preprocess(reference_sentence) for reference_sentence in reference_sentences])

        for i in range(len(generate_corpus)):
            curMatchWordCount = defaultdict(float)
            curRefWordCount = defaultdict(float)
            curGenWordCount = defaultdict(float)
            curMatchCharCount = defaultdict(float)
            curRefCharCount = defaultdict(float)
            curGenCharCount = defaultdict(float)

            generate_char_ngrams = self._generate_ngrams(input_sentence=generate_corpus_process[i], task_type='char')
            generate_word_ngrams = self._generate_ngrams(input_sentence=generate_corpus[i], task_type='word')
            
            cur_max_F = 0
            for j in range(len(reference_corpus[i])):
                reference_char_ngrams = self._generate_ngrams(input_sentence=reference_corpus_process[i][j], task_type='char')
                reference_word_ngrams = self._generate_ngrams(input_sentence=reference_corpus[i][j], task_type='word')

                matchNgramWordCount, totalGenNgramWordCount, totalRefNgramWordCount = self._ngrams_match(gen_ngrams=generate_word_ngrams, ref_ngrams=reference_word_ngrams)
                matchNgramCharCount, totalGenNgramCharCount, totalRefNgramCharCount = self._ngrams_match(gen_ngrams=generate_char_ngrams, ref_ngrams=reference_char_ngrams)

                ngramWordF, _, _ = self._calc_F(matchNgramWordCount, totalGenNgramWordCount, totalRefNgramWordCount, beta=self.beta)
                ngramCharF, _, _ = self._calc_F(matchNgramCharCount, totalGenNgramCharCount, totalRefNgramCharCount, beta=self.beta)
            
                cur_F = (sum(ngramCharF.values()) + sum(ngramWordF.values())) / (max(self.char_n_grams) + max(self.word_n_grams))

                if cur_F > cur_max_F:
                    cur_max_F = cur_F
                    curMatchWordCount = matchNgramWordCount
                    curRefWordCount = totalRefNgramWordCount
                    curGenWordCount = totalGenNgramWordCount
                    curMatchCharCount = matchNgramCharCount
                    curRefCharCount = totalRefNgramCharCount
                    curGenCharCount = totalGenNgramCharCount
                
            for ngram in self.char_n_grams:
                totalMatchCharCount[ngram] += curMatchCharCount[ngram]
                totalRefCharCount[ngram] += curRefCharCount[ngram]
                totalGenCharCount[ngram] += curGenCharCount[ngram]

            for ngram in self.word_n_grams:
                totalMatchWordCount[ngram] += curMatchWordCount[ngram]
                totalRefWordCount[ngram] += curRefWordCount[ngram]
                totalGenWordCount[ngram] += curGenWordCount[ngram]
            
            avgTotalF += cur_max_F

        totalWordF, totalWordRecall, totalWordPrec = self._calc_F(totalMatchWordCount, totalGenWordCount, totalRefWordCount, beta=self.beta)
        totalCharF, totalCharRecall, totalCharPrec = self._calc_F(totalMatchCharCount, totalGenCharCount, totalRefCharCount, beta=self.beta)
        
        totalF = (sum(totalCharF.values()) + sum(totalWordF.values())) / (max(self.char_n_grams) + max(self.word_n_grams))
        totalRecall = (sum(totalCharRecall.values()) + sum(totalWordRecall.values())) / (max(self.char_n_grams) + max(self.word_n_grams))
        totalPrec = (sum(totalCharPrec.values()) + sum(totalWordPrec.values())) / (max(self.char_n_grams) + max(self.word_n_grams))
        avgTotalF /= len(generate_corpus)

        result['precision'] = totalPrec
        result['recall'] = totalRecall
        result['document-F'] = totalF
        result['avg-sentence-F'] = avgTotalF
        result['beta'] = self.beta
        return result

    def __str__(self):
        mesg = 'The CHRF++ Evaluator Info:\n' + '\tMetrics:[CHRF, WORDF], Char_Ngram:[' + ', '.join(map(str, self.char_n_grams)) + \
               '], Word_Ngram:[' + ', '.join(map(str, self.word_n_grams)) + ']'
        return mesg