# @Time   : 2021/4/15
# @Author : Tianyi Tang
# @Email  : steven_tang@ruc.edu.cn

import argparse
from textbox.utils import init_logger, get_model, get_trainer
from textbox.config import Config
from textbox.data import data_preparation


def run_textbox(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """

    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)

    print(f"Please input the sentence you want to conduct {config['task_type']}:")
    sentence = input().strip()

    # dataset splitting
    test_data = data_preparation(config, sentence)

    # model loading and initialization
    model = get_model(config['model'])(config, test_data).to(config['device'])
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    print('\n')
    print(f"The {config['task_type']} result:")
    trainer.evaluate(test_data, load_best_model=saved, model_file=config['load_experiment'], eval=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='RNN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='COCO', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_textbox(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict={
            'state': 'critical',
            'test_only': True
        }
    )
