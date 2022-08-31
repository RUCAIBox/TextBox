import argparse
from textbox import run_textbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BART', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='samsum', help='name of datasets')
    parser.add_argument('--config_files', type=str, nargs='*', default=list(), help='config files')

    args, _ = parser.parse_known_args()

    run_textbox(model=args.model, dataset=args.dataset, config_file_list=args.config_files, config_dict={})
