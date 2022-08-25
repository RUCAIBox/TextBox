#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

# wrapper to parse text file with StanfordCoreNLP tools and print it in CoNLL format
# Input must be tokenized, and one line per sentence.

# requirements:
# - Stanford CoreNLP
# - English models for CoreNLP

from __future__ import print_function, unicode_literals
import os
import sys
import codecs
import io
import argparse

from collections import defaultdict
from subprocess import Popen, PIPE

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--java', type=str, help = "path to JAVA runtime binary", default = 'java')
    parser.add_argument('--corenlp', type=str, required=True, help = "path to stanford-corenlp-{version}.jar file")
    parser.add_argument('--corenlp-models', type=str, required=True, help = "path to stanford-corenlp-{version}-models.jar")

    return parser.parse_args()

def process_stanford(infile, java, corenlp, corenlp_models):

    stanford = Popen([java,
               '-cp', corenlp + ':' + corenlp_models,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP',
               '-annotators', 'tokenize, ssplit, pos, depparse, lemma',
               '-ssplit.eolonly', 'true',
               '-tokenize.whitespace', 'true',
               '-numThreads', '8',
               '-textFile', '-',
               'outFile', '-'], stdin=infile, stdout = PIPE, stderr = open('/dev/null', 'w'))
    return stanford.stdout


def get_sentences(instream):
    sentence = []
    expect = 0

    for line in instream:
        if expect == 0 and line.startswith('Sentence #'):
            if sentence:
                yield sentence
            sentence = []
            expect = 1

        elif line == '\n':
            expect = 0

        elif expect == 3:
            try:
                rel, remainder = line.split('(')
            except:
                sys.stderr.write(line + '\n')
                raise
            head, dep = remainder.split()
            head_int = int(head.split('-')[-1][:-1])
            dep_int = int(dep.split('-')[-1][:-1])
            sentence[dep_int-1]['head'] = head_int
            sentence[dep_int-1]['label'] = rel

        elif expect == 2:
            linesplit = line.split('[',1)[1].rsplit(']',1)[0].split('] [')
            if len(linesplit) != len(sentence):
                sys.stderr.write('Warning: mismatch in number of words in sentence\n')
                sys.stderr.write(' '.join(w['word'] for w in sentence))
                for i in range(len(sentence)):
                    sentence[i]['pos'] = '-'
                    sentence[i]['lemma'] = '-'
                    sentence[i]['head'] = 0
                    sentence[i]['label'] = '-'
                expect = 0
                continue
            for i,w in enumerate(linesplit):
                sentence[i]['pos'] = w.split(' PartOfSpeech=')[-1].split()[0]
                sentence[i]['lemma'] = w.split(' Lemma=')[-1]
            expect = 3

        elif expect == 1:
            for w in line.split():
                sentence.append({'word':w})
            expect = 2

    if sentence:
        yield sentence

def write(sentence):
    for i, w in enumerate(sentence):
      sys.stdout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(i+1, w['word'], w['lemma'], w['pos'], w['pos'], '-', w['head'], w['label']))

if __name__ == '__main__':
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

    args = parse_args()

    stanford = process_stanford(sys.stdin, args.java, args.corenlp, args.corenlp_models)
    for sentence in get_sentences(codecs.getreader('UTF-8')(stanford)):
       write(sentence)
       sys.stdout.write('\n')
