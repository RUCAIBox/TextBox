import torch
from logging import getLogger
from accelerate import Accelerator
from accelerate.utils import set_seed
from textbox.utils.logger import init_logger
from textbox.utils.utils import get_model, get_tokenizer, get_trainer, init_seed
from textbox.config.configurator import Config
from textbox.data.utils import data_preparation
from textbox.utils.dashboard import init_dashboard, start_dashboard, finish_dashboard


def run_textbox(model=None, dataset=None, config_file_list=None, config_dict=None):
    from textbox.quick_start.experiment import Experiment
    experiment = Experiment(model, dataset, config_file_list, config_dict)
    experiment.run(do_train=True, do_valid=True, do_test=True)



def _run_textbox(model=None, dataset=None, config_file_list=None, config_dict=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
    """

    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    accelerator = Accelerator()
    config['device'] = accelerator.device
    config['is_local_main_process'] = accelerator.is_local_main_process

    # reproducibility initialization
    init_seed(config['seed'], config['reproducibility'])
    set_seed(config['seed'])

    # logger initialization
    init_logger(config['filename'], config['state'], accelerator.is_local_main_process, config['logdir'])
    init_dashboard(config)
    logger = getLogger()
    logger.info(config)
    # ================ ROUND 1 ================
    start_dashboard()

    # dataset initialization
    tokenizer = get_tokenizer(config)
    train_data, valid_data, test_data = data_preparation(config, tokenizer)
    train_data, valid_data, test_data = accelerator.prepare(train_data, valid_data, test_data)

    # model loading and initialization
    model = get_model(config['model_name'])(config, tokenizer).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['model'])(config, model, accelerator)

    if config['test_only']:
        # test only
        logger.info('Test only')
        if not config['load_experiment']:
            logger.warning('Specific path to model file with `load_experiment`.')
        test_result = trainer.evaluate(test_data, model_file=config['load_experiment'])
    else:
        # checkpoint initialization
        if config['load_experiment'] is not None:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        # do_train & do_test
        result = trainer.fit(train_data, valid_data)
        # do_eval
        for key, value in result.items():
            logger.info(f"{key}: {value}")
        test_result = trainer.evaluate(test_data)

    # finish
    logger.info('test result: {}'.format(test_result))
    finish_dashboard()

    # ================ ROUND 2 ================
    start_dashboard()
    # dataset initialization
    tokenizer = get_tokenizer(config)
    train_data, valid_data, test_data = data_preparation(config, tokenizer)
    train_data, valid_data, test_data = accelerator.prepare(train_data, valid_data, test_data)

    # model loading and initialization
    model = get_model(config['model_name'])(config, tokenizer).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['model'])(config, model, accelerator)
    if config['test_only']:
        # test only
        logger.info('Test only')
        if not config['load_experiment']:
            logger.warning('Specific path to model file with `load_experiment`.')
        test_result = trainer.evaluate(test_data, model_file=config['load_experiment'])
    else:
        # checkpoint initialization
        if config['load_experiment'] is not None:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        # do_train & do_test
        result = trainer.fit(train_data, valid_data)
        # do_eval
        for key, value in result.items():
            logger.info(f"{key}: {value}")
        test_result = trainer.evaluate(test_data)

    # finish
    logger.info('test result: {}'.format(test_result))
    finish_dashboard()
