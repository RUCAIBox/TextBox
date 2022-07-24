from logging import getLogger, Logger
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from textbox.trainer.trainer import AbstractTrainer, Trainer
from textbox.utils.logger import init_logger
from textbox.utils.utils import get_model, get_tokenizer, get_trainer, init_seed
from textbox.config.configurator import Config
from textbox.data.utils import data_preparation
from textbox.utils.dashboard import init_dashboard, finish_dashboard

from typing import Optional, Tuple, Any, List, Dict
ResultType = Dict[str, Any]


class Experiment:
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
    """

    def __init__(
            self,
            model: Optional[str] = None,
            dataset: Optional[str] = None,
            config_file_list: Optional[List[str]] = None,
            config_dict: Optional[Dict[str, Any]] = None,
    ):

        self.accelerator = Accelerator()

        # for safety reasons, a direct modification of config is not suggested
        if not isinstance(config_dict, dict):
            config_dict = dict()
        config_dict.update({
            'is_local_main_process': self.accelerator.is_local_main_process,
        })
        self.config = self.init_config(model, dataset, config_file_list, config_dict)
        self.logger = getLogger()

        init_dashboard(self.config)
        init_seed(self.config['seed'], self.config['reproducibility'])
        set_seed(self.config['seed'])
        self.train_data, self.valid_data, self.test_data, self.tokenizer = \
            self._init_data(self.config, self.accelerator)
        self.model = get_model(self.config['model_name'])(self.config, self.tokenizer).to(self.config['device'])
        self.logger.info(self.model)
        self.trainer: Trainer = get_trainer(self.config['model'])(self.config, self.model, self.accelerator)
        # reproducibility initialization
        self.do_train = True if self.config['do_train'] is None else self.config['do_train']
        self.do_valid = True if self.config['do_valid'] is None else self.config['do_valid']
        self.do_test = True if self.config['do_test'] is None else self.config['do_test']
        self.valid_result: Optional[ResultType] = None
        self.test_result: Optional[ResultType] = None

    @staticmethod
    def init_config(
            model: Optional[str] = None,
            dataset: Optional[str] = None,
            config_file_list: Optional[List[str]] = None,
            config_dict: Optional[Dict[str, Any]] = None,
    ) -> Config:

        # configurations initialization
        config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)

        # logger initialization
        init_logger(config['filename'], config['state'], config['is_local_main_process'], config['logdir'])
        logger = getLogger()
        logger.info(config)

        return config

    @staticmethod
    def _init_data(config: Config, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
        tokenizer = get_tokenizer(config)
        train_data, valid_data, test_data = data_preparation(config, tokenizer)
        train_data, valid_data, test_data = accelerator.prepare(train_data, valid_data, test_data)
        return train_data, valid_data, test_data, tokenizer

    def _do_train_and_valid(self):

        if not self.do_train and self.do_valid:
            raise ValueError('Cannot execute validation without training.')

        if self.do_train:
            if self.config['load_experiment'] is not None:
                self.trainer.resume_checkpoint(resume_file=self.config['load_experiment'])
            train_data = self.train_data
            valid_data = self.valid_data if self.do_valid else None

            self.valid_result = self.trainer.fit(train_data, valid_data)

            self.logger.info('test result: {}'.format(self.valid_result))

    def _do_test(self):

        if self.do_test:

            self.test_result = self.trainer.evaluate(self.test_data, model_file=self.config['load_experiment'])

            for key, value in self.test_result.items():
                self.logger.info(f"{key}: {value}")

    def _on_experiment_end(self):
        finish_dashboard()

    def run(self) -> Tuple[Optional[ResultType], Optional[ResultType]]:

        self._do_train_and_valid()
        self._do_test()

        self._on_experiment_end()
        return self.valid_result, self.test_result
