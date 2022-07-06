# @Time   : 2020/11/5
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

import argparse

from textbox.quick_start import run_textbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='RNN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='COCO', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_textbox(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict={})
