# @Time   : 2020/11/14
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

# UPDATE:
# @Time   : 2021/4/12
# @Author : Lai Xu
# @Email  : tsui_lai@163.com

"""
textbox.evaluator.distinct_evaluator
#######################################
"""

import numpy as np

class DistinctEvaluator():
    r"""Distinct Evaluator. Now, we support metrics `'distinct'`.
    """

    def __init__(self, config):
        self.n_grams = config['n_grams']
    
    def evaluate(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus
        
        Returns:
            dict: such as ``{'distinct-1': xxx}``
        """
        metric_dict = {}
        dist_dict = self._calc_dist_info(generate_corpus=generate_corpus)
        for key in dist_dict:
            tp_val = dist_dict[key]
            metric_dict[key] = round(tp_val, 4)
        return metric_dict
    
    def dist_func(self, generate_corpus, ngram):
        ngram_pool = []
        for sent in generate_corpus:
            tokens = sent[:ngram]
            ngram_pool.append(" ".join(tokens))
            for i in range(1, len(sent) - ngram + 1):
                tokens = tokens[1:]
                tokens.append(sent[i + ngram - 1])
                ngram_pool.append(" ".join(tokens))
        return len(set(ngram_pool)) / len(ngram_pool)
    
    def _calc_dist_info(self, generate_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus

        Returns:
            list: a list of metrics <metric> which record the results
        """
        dist_dict = {}
        for n_gram in self.n_grams:
            result = self.dist_func(generate_corpus=generate_corpus, ngram=n_gram)
            key = "distinct-{}".format(n_gram)
            dist_dict[key] = result
        return dist_dict
    
    def __str__(self):
        mesg = "The Distinct Evaluator Info:\n" + "\tMetrics:[distinct], Ngram:["\
                + ", ".join(map(str, self.n_grams)) + "]"
        return mesg