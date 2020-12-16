# @Time   : 2020/11/5
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/3
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/8
# @Author : Gaole He
# @Email  : hegaole@ruc.edu.cn

"""
textbox.quick_start
########################
"""
import logging
from logging import getLogger
from textbox.utils import init_logger, get_model, get_trainer, init_seed
from textbox.config import Config
from textbox.data import create_dataset, data_preparation


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

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    if config['test_only']:
        logger.info('Test only')
        test_result = trainer.evaluate(test_data, load_best_model=saved, model_file=config['load_experiment'])
    else:
        if config['load_experiment'] is not None:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=saved)

        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=saved)

        logger.info('best valid loss: {}, best valid ppl: {}'.format(best_valid_score, best_valid_result))

    logger.info('test result: {}'.format(test_result))

def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict): parameters dictionary used to modify experiment parameters
        config_file_list (list): config files used to modify experiment parameters
        saved (bool): whether to save the model
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
