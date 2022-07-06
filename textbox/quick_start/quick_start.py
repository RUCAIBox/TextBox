import torch
from logging import getLogger
from textbox.utils import init_logger, get_model, get_trainer, init_seed
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

    local_rank = None
    if config['DDP']:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        config['device'] = torch.device("cuda", local_rank)

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    is_logger = (config['DDP'] and torch.distributed.get_rank() == 0) or not config['DDP']

    init_logger(config['filename'], config['state'], is_logger)
    logger = getLogger()
    logger.info(config)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config)

    # model loading and initialization
    single_model = get_model(config['model'])(config, train_data).to(config['device'])
    if config['DDP']:
        if config['find_unused_parameters']:
            model = torch.nn.parallel.DistributedDataParallel(
                single_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                single_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )
    else:
        model = single_model

    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['model'])(config, model)

    if config['test_only']:
        logger.info('Test only')
        if not config['load_experiment']:
            logger.warning('Specific path to model file with `load_experiment`.')
        test_result = trainer.evaluate(test_data, load_best_model=saved, model_file=config['load_experiment'])
    else:
        if config['load_experiment'] is not None and is_logger:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        # model training
        result = trainer.fit(train_data, valid_data, saved=saved)
        # model evaluating
        if config['DDP']:
            if torch.distributed.get_rank() != 0:
                return
            config['DDP'] = False
            model = get_model(config['model'])(config, train_data).to(config['device'])
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        for key, value in result.items():
            logger.info(f"{key}: {value}")
        test_result = trainer.evaluate(test_data, load_best_model=saved)

    logger.info('test result: {}'.format(test_result))
