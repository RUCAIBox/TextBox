import argparse
from textbox import run_hyper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyper tuning")
    parser.add_argument('--model', '-m', type=str, default='BART', help='name of model')
    parser.add_argument('--dataset', '-d', type=str, default='samsum', help='name of dataset')
    parser.add_argument('--config_files', type=str, nargs='*', default=list(), help='config files')
    parser.add_argument('--params_file', type=str, default=None, help='path to file containing parameters to be tuned')
    parser.add_argument('--output_file', type=str, default='hyper_example.result', help='path to hyper-tuning output')

    args, _ = parser.parse_known_args()

    print(args.config_files)
    run_hyper(
        model=args.model,
        dataset=args.dataset,
        base_config_file_list=args.config_files,
        base_config_dict={},
        space_file=args.params_file,
        path_to_output=args.output_file,
    )
