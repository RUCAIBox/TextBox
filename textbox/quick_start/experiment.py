import os
import logging
from copy import copy
from logging import getLogger
from typing import Optional, Tuple, Any, List, Dict

from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..config.configurator import Config
from ..data.utils import data_preparation
from ..trainer.trainer import Trainer
from ..utils.dashboard import SummaryTracker
from ..utils.logger import init_logger
from ..utils.utils import get_model, get_tokenizer, get_trainer, init_seed

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
        self.__base_config = Config(model, dataset, config_file_list, config_dict)
        self.__extended_config = None

        self.accelerator = Accelerator()
        self.__base_config.update({'_is_local_main_process': self.accelerator.is_local_main_process})
        self.logger = self.init_logger(self.__base_config)
        self.summary_tracker = SummaryTracker.basicConfig(self.get_config())
        self.train_data, self.valid_data, self.test_data, self.tokenizer = \
            self._init_data(self.get_config(), self.accelerator)

    def get_config(self) -> Config:
        config = copy(self.__base_config)
        if self.__extended_config is not None:
            config.update(self.__extended_config)
        return config

    @staticmethod
    def init_logger(config: Config) -> logging.Logger:

        # logger initialization
        init_logger(
            filename=config['filename'],
            log_level=config['state'],
            enabled=config['_is_local_main_process'],
            logdir=config['logdir'],
            total_dir=config['total_dir']
        )
        logger = getLogger(__name__)
        logger.info(config)

        return logger

    @staticmethod
    def _init_data(config: Config, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
        tokenizer = get_tokenizer(config)
        train_data, valid_data, test_data = data_preparation(config, tokenizer)
        train_data, valid_data, test_data = accelerator.prepare(train_data, valid_data, test_data)
        return train_data, valid_data, test_data, tokenizer

    def _on_experiment_start(self, extended_config: Optional[dict]):
        """(Re-)initialize configuration. Since for now config and trainer is modifiable, this
        function is needed to ensure they were aligned to initial configuration.
        """
        self.__extended_config = extended_config
        config = self.get_config()
        init_seed(config['seed'], config['reproducibility'])

        self.model = get_model(config['model'])(config, self.tokenizer).to(config['device'])
        self.logger.info(self.model)
        self.trainer: Trainer = get_trainer(config['model'])(config, self.model, self.accelerator)

        self.do_train = config['do_train']
        self.do_valid = config['do_valid']
        self.do_test = config['do_test']
        self.valid_result: Optional[ResultType] = None
        self.test_result: Optional[ResultType] = None

    def _do_train_and_valid(self):

        if not self.do_train and self.do_valid:
            raise ValueError('Cannot execute validation without training.')

        if self.do_train:
            if self.__base_config['load_experiment'] is not None:
                self.trainer.resume_checkpoint(resume_file=self.__base_config['load_experiment'])
            train_data = self.train_data
            valid_data = self.valid_data if self.do_valid else None

            self.valid_result = self.trainer.fit(train_data, valid_data)

    def _do_test(self):
        if self.do_test:
            with self.summary_tracker.new_epoch('eval'):
                self.test_result = self.trainer.evaluate(
                    self.test_data, model_file=self.__base_config['load_experiment']
                )
                self.summary_tracker.set_metrics_results(self.test_result)
                self.logger.info(
                    'Evaluation result:\n{}'.format(self.summary_tracker._current_epoch.as_str(sep=",\n", indent=" "))
                )

    def _on_experiment_end(self):
        if self.__base_config['max_save'] == 0:
            saved_filename = os.path.abspath(
                os.path.join(self.__base_config['checkpoint_dir'], self.__base_config['filename']) + '.pth'
            )
            saved_link = os.readlink(saved_filename) if os.path.exists(saved_filename) else ''
            from ..utils import safe_remove
            safe_remove(saved_filename)
            safe_remove(saved_link)
        self.__extended_config = None

    def run(self, extended_config: Optional[dict] = None):
        with self.summary_tracker.new_experiment():
            self._on_experiment_start(extended_config)
            self._do_train_and_valid()
            self._do_test()
            self._on_experiment_end()
