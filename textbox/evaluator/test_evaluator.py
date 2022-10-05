# from distutils.command.config import config
# from bleu_evaluator import BleuEvaluator
# from nist_evaluator import NistEvaluator
# from spice_evaluator import SpiceEvaluator
# from cider_evaluator import CiderEvaluator
from meteor_evaluator import MeteorEvaluator
from rouge_evaluator import RougeEvaluator
# from chrf_evaluator import ChrfEvaluator
# from ter_evaluator import TerEvaluator
# from distinct_evaluator import DistinctEvaluator
# from unique_evaluator import UniqueEvaluator
# from selfbleu_evaluator import SelfBleuEvaluator
# from bertscore_evaluator import BertScoreEvaluator
# from qa_evaluator import QaEvaluator


def _proc(l):
    if len(l) >= 2 and ((l[0] == '"' and l[-1] == '"') or (l[0] == "'" and l[-1] == "'") or
                        (l[0] == '[' and l[-1] == ']')):
        try:
            l = eval(l)
            if not isinstance(l, list):
                l = str(l)
        except:
            pass
    return l


config = {
    'bleu_max_ngrams': 4,
    'bleu_type': 'multi-bleu',
    'rouge_type': 'py-rouge',
    'multiref_strategy': 'leave_one_out',
    'rouge_max_ngrams': 2,
    'meteor_type': 'pycocoevalcap',
    'chrf_type': 'sacrebleu',
    'smoothing_function': 0,
    'corpus_bleu': True,
    'corpus_meteor': True,
    'dataset': 'pc',
    'lower': True,
    'filename': 'tty',
    'distinct_max_ngrams': 4,
    'inter_distinct': True,
    'unique_max_ngrams': 4,
    'self_bleu_max_ngrams': 4,
    'bert_score_model_type': None,
    'tgt_lang': 'en',
    'device': 'cuda',
    'eval_batch_size': 64
}
# bleu = BleuEvaluator(config)
# spice = SpiceEvaluator(config)
# cider = CiderEvaluator(config)
meteor = MeteorEvaluator(config)
rouge = RougeEvaluator(config)
# chrf = ChrfEvaluator(config, 'chrf++')
# ter = TerEvaluator(config)
# distinct = DistinctEvaluator(config)
# unique = UniqueEvaluator(config)
# self_bleu = SelfBleuEvaluator(config)
# bert_score = BertScoreEvaluator(config)
# qa = QaEvaluator(config)

gen = [
    'It is a guide to action which ensures that the military always obeys the commands of the party',
    'he read the book because he was interested in world history'
]
ref = [['It is a guide to action that ensures that the military will forever heed Party commands'],
       ['he was interested in world history because he read the book']]
ref = [[
    'It is a guide to action that ensures that the military will forever heed Party commands',
    'It is the guiding principle which guarantees the military forces always being under the command of the Party',
    'It is the practical guide for the army always to heed the directions of the party'
], ['he was interested in world history because he read the book']]
'''
gen = [r.strip().lower() for r in open('/home/tangtianyi/ICML/generated/BART-webnlg-Apr-18-2022_18-56-52.txtbest').readlines()]
ref = [[]]
for line in open('/home/tangtianyi/ICML/dataset/webnlg/test.tgt', 'r'):
    line = line.strip().lower()
    line = _proc(line)
    if line:
        ref[-1].append(line)
    else:
        ref.append([])
if not ref[-1]:
    del ref[-1]
'''
'''
gen = [r.strip().lower() for r in open('/home/tangtianyi/ICML/generated/BART-da-May-25-2022_09-43-08.txtbest').readlines()]
ref = [_proc(r.strip().lower()).split(' | ') for r in open('/home/tangtianyi/ICML/dataset/da/test.tgt').readlines()]
'''
'''
gen = [r.strip().lower() for r in open('/home/tangtianyi/ICML/dataset/SQuAD/gen.txt').readlines()]
ref = [[]]
for line in open('/home/tangtianyi/ICML/dataset/SQuAD/test_new.tgt', 'r'):
    line = line.strip().lower()
    line = _proc(line)
    if line:
        ref[-1].append(line)
    else:
        ref.append([])
if not ref[-1]:
    del ref[-1]
print(len(gen), len(ref))
'''
#gen = [r.strip() for r in open('/home/tangtianyi/ICML/generated/BART-totto-May-25-2022_09-30-38.txtbest').readlines()]
#ref = [_proc(r.strip()) for r in open('/home/tangtianyi/ICML/dataset/totto/test.tgt').readlines()]
# gen = [r.strip().lower() for r in open('/home/tangtianyi/TextBox/generated/BART-samsum-2022-Jul-08_10-24-17.txt')]
# ref = [[r.strip().lower()] for r in open('/home/tangtianyi/ICML/dataset/samsum/test.tgt')]

# gen = [r.strip().lower() for r in open('/home/tangtianyi/ICML/generated/BART-PC-May-31-2022_22-56-56.txtbest')]
gen = [r.strip() for r in open('/mnt/tangtianyi/AESOP/evaluation/bart0-14_sep_extract')]
ref = [[_proc(r.strip())] for r in open('/mnt/tangtianyi/AESOP/evaluation/quora/test.ref').readlines()]
# gen = [r.strip() for r in open('/home/tangtianyi/ICML/generated/BART-squad-May-25-2022_22-15-11.txtbest')]
# ref = [_proc(r.strip()) for r in open('/home/tangtianyi/ICML/dataset/squad/test.tgt').readlines()]
print(meteor.evaluate(gen, ref))
