import argparse
from textbox import run_hyper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyper tuning")
    parser.add_argument('--model', '-m', type=str, default='BART', help='name of model')
    parser.add_argument('--dataset', '-d', type=str, default='samsum', help='name of dataset')
    parser.add_argument('--config_files', type=str, nargs='*', default=list(), help='config files')
    parser.add_argument('--space', type=str, default=None, help='path to file containing parameters to be tuned')
    parser.add_argument('--algo', type=str, default='exhaustive', help='algorithm used')

    args, _ = parser.parse_known_args()

    run_hyper(
        algo=args.algo,
        model=args.model,
        dataset=args.dataset,
        base_config_file_list=args.config_files,
        base_config_dict={},
        space=args.space,
    )
