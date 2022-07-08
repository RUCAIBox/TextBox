import torch
import logging
from logging import getLogger
from textbox import Config, data_preparation
from textbox.utils import init_logger, get_tokenizer, get_model, get_trainer, init_seed


def run_textbox(model=None, dataset=None, config_file_list=None, config_dict=None):
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

    if config['DDP']:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        config['device'] = torch.device("cuda", local_rank)

    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    is_logger = (config['DDP'] and torch.distributed.get_rank() == 0) or not config['DDP']

    if is_logger:
        init_logger(config)
        logger = getLogger()
        logger.info(config)
        logger.setLevel(logging.INFO)

    tokenizer = get_tokenizer(config)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, tokenizer)

    # model loading and initialization
    single_model = get_model(config['model'])(config, tokenizer).to(config['device'])
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

    if is_logger:
        logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['model'])(config, model)

    if config['test_only']:
        logger.info('Test only')
        test_result = trainer.evaluate(test_data, model_file=config['load_experiment'])
    else:
        if config['load_experiment'] is not None and is_logger:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        if (config['DDP'] == True):
            if (torch.distributed.get_rank() != 0):
                return
            config['DDP'] = False
            model = get_model(config['model'])(config, train_data).to(config['device'])
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        logger.info('best valid loss: {}, best valid ppl: {}'.format(best_valid_score, best_valid_result))
        test_result = trainer.evaluate(test_data)

    logger.info('test result: {}'.format(test_result))
