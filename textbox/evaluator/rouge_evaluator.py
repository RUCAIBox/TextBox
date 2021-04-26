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
textbox.evaluator.rouge_evaluator
#################################
"""
import os
import tempfile
import logging
from files2rouge import settings
from files2rouge import utils
from pyrouge import Rouge155
from collections import defaultdict
from textbox.evaluator.abstract_evaluator import AbstractEvaluator

class RougeEvaluator(AbstractEvaluator, Rouge155):
    r"""Rouge Evaluator. Now we support rouge-based ngram metrics which conains rouge-n, rouge-l and rouge-w.
    """
    def __init__(self):
        #"-c 95 -r 1000 -n 2 -w 1.2 -a"
        self.rouge_args =[
                '-c', 95,
                '-r', 1000,
                '-n', 2,
                '-w', 1.2,
                '-a'
                ]
        self._deal_args()

    def _deal_args(self):
        self.rouge_args = " ".join([str(_) for _ in self.rouge_args])

    def _preprocess(self, input_sentence):
        return " ".join(input_sentence)

    def _write_file(self, write_path, content):
        f = open(write_path, 'w')
        f.write("\n".join(content))
        f.close()

    def _split_rouge(self, input_sentence):
        res_list = input_sentence.split()
        res = {}
        res[res_list[1].lower()] = float(res_list[3])
        return res

    def _calc_rouge(self, args):
        summ_path = args['summ_path']
        ref_path = args['ref_path']
        eos = args['eos']
        ignore_empty_reference = args['ignore_empty_reference']
        ignore_empty_summary = args['ignore_empty_summary']
        stemming = args['stemming']

        s = settings.Settings()
        s._load()
        with tempfile.TemporaryDirectory() as dirpath:  # generate virtual route
            sys_root, model_root = [os.path.join(dirpath, _) for _ in ["system", "model"]]
            utils.mkdirs([sys_root, model_root])
            ignored = utils.split_files(model_path=ref_path,
                                        system_path=summ_path,
                                        model_dir=model_root,
                                        system_dir=sys_root,
                                        eos=eos,
                                        ignore_empty_reference=ignore_empty_reference,
                                        ignore_empty_summary=ignore_empty_summary)
            r = Rouge155(rouge_dir=os.path.dirname(s.data['ROUGE_path']),
                                log_level=logging.ERROR,
                                stemming=stemming)
            r.system_dir = sys_root
            r.model_dir = model_root
            r.system_filename_pattern = r's.(\d+).txt'
            r.model_filename_pattern = 'm.[A-Z].#ID#.txt'
            data_arg = "-e %s" % s.data['ROUGE_data']
            rouge_args_str = "%s %s" % (data_arg, self.rouge_args)
            output = r.convert_and_evaluate(rouge_args=rouge_args_str)
            res = self._get_info(output)
        return res

    def _get_info(self, input_str):
        rouge_list = input_str.replace("---------------------------------------------", "").replace("\n\n", "\n").strip().split("\n")
        rouge_list = [rouge for rouge in rouge_list if "Average_F" in rouge]
        rouge_dict = defaultdict(float)
        for each in list(map(self._split_rouge, rouge_list)):
            rouge_dict.update(each)
        return rouge_dict

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        generate_corpus = [self._preprocess(generate_sentence) for generate_sentence in generate_corpus]
        reference_corpus = [self._preprocess(reference_sentence) for reference_sentence in reference_corpus]
        with tempfile.TemporaryDirectory() as path:
            generate_path = os.path.join(path, 'generate_corpus.txt')
            reference_path = os.path.join(path, 'reference_corpus.txt')
            self._write_file(generate_path, generate_corpus)
            self._write_file(reference_path, reference_corpus)
            
            calc_args = {
                'summ_path': generate_path,
                'ref_path': reference_path,
                'eos': '.',
                'ignore_empty_reference': False,
                'ignore_empty_summary': False,
                'stemming': True
            }
            res = self._calc_rouge(calc_args)
        return res