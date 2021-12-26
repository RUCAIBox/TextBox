import pickle
import os
import collections
import sys

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor


class Kb2Text_evaluator(object):
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]

    def convert(self, data):
        if isinstance(data, basestring):
            return data.encode('utf-8')
        elif isinstance(data, collections.Mapping):
            return dict(map(convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(convert, data))
        else:
            return data

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, get_scores=True, live=False, **kwargs):
        if live:
            temp_ref = kwargs.pop('ref', {})
            cand = kwargs.pop('cand', {})
        else:
            reference_path = kwargs.pop('ref', '')
            candidate_path = kwargs.pop('cand', '')

            # load caption data
            with open(reference_path, 'rb') as f:
                temp_ref = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                cand = pickle.load(f)

        # make dictionary
        hypo = {}
        ref = {}
        i = 0
        for vid, caption in cand.items():
            hypo[i] = [caption]
            ref[i] = temp_ref[vid]
            i += 1

        # compute scores
        final_scores = self.score(ref, hypo)
        # """
        # print out scores
        print('Bleu_1:\t', final_scores['Bleu_1'])
        print('Bleu_2:\t', final_scores['Bleu_2'])
        print('Bleu_3:\t', final_scores['Bleu_3'])
        print('Bleu_4:\t', final_scores['Bleu_4'])
        print('METEOR:\t', final_scores['METEOR'])
        print('ROUGE_L:', final_scores['ROUGE_L'])
        # print ('CIDEr:\t', final_scores['CIDEr'])
        # """

        if get_scores:
            return final_scores
