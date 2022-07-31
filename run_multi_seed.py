import argparse
from textbox import run_multi_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multiple random seed test.")
    parser.add_argument('--model', '-m', type=str, default='BART', help='name of model')
    parser.add_argument('--dataset', '-d', type=str, default='samsum', help='name of dataset')
    parser.add_argument('--config_files', type=str, nargs='*', default=list(), help='config files')
    parser.add_argument('--multi_seed', type=int, help='the amount of random seed', required=True)

    args, _ = parser.parse_known_args()

    run_multi_seed(
        multi_seed=args.multi_seed,
        model=args.model,
        dataset=args.dataset,
        base_config_file_list=args.config_files,
        base_config_dict={},
    )
