import collections
import os
from logging import getLogger
from typing import Optional, Union, List, Tuple, Dict

import torch
import torch.optim as optim
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from textbox import Config
from textbox.utils.dashboard import get_dashboard, Timestamp, EpochTracker
from .scheduler import (
    AbstractScheduler, InverseSquareRootScheduler, CosineScheduler, LinearScheduler, ConstantScheduler
)
from ..data.abstract_dataloader import AbstractDataLoader
from ..evaluator import BaseEvaluator
from ..model.abstract_model import AbstractModel
from ..utils import ensure_dir, serialized_save, init_seed


class AbstractTrainer:
    r"""Trainer Class is used to manage the training and evaluation processes of text generation system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config: Config, model: AbstractModel):
        self.config = config
        self.model = model
        self.logger = getLogger(__name__)

    def fit(self, train_data: AbstractDataLoader):
        r"""Train the model based on the train data.
        """

        raise NotImplementedError('Method `fit()` should be implemented.')

    def evaluate(self, eval_data: AbstractDataLoader):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method `evaluate()` should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in text generation systems.

    This class defines common functions for training and evaluation processes of most text generation system models,
    including `fit()`, `evaluate()`, `resume_checkpoint()` and some other features helpful for model training and
    evaluation.

    Generally speaking, this class can serve most text generation system models, If the training process of the model
    is to simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters' information
    for controlling training and evaluation, such as `learning_rate`, `epochs` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.
    """

    def __init__(self, config: Config, model: AbstractModel, accelerator: Accelerator):
        super(Trainer, self).__init__(config, model)

        self.device: torch.device = config['device']
        self.filename = self.config['filename']
        self.is_chinese_task = self.config['is_chinese_task']
        self.accelerator = accelerator

        # Optimization strategy
        self.learning_rate = config['learning_rate']
        self.optimizer_kwargs = {'lr': config['learning_rate']}
        self.optimizer_kwargs.update(config['optimizer_kwargs'])
        self.adafactor_kwargs = config['adafactor_kwargs']
        self.scheduler_kwargs = config['scheduler_kwargs']
        self.grad_clip = config['grad_clip']
        self._trainable_parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = self._build_optimizer(config['optimizer'], config['scheduler'])
        self.accumulation_steps = config['accumulation_steps']

        # Training strategy
        self.quick_test = bool(config['quick_test'])
        self.start_epoch = 0
        r"""Start epoch index. That is, `epoch_idx` iterates through `range(self.start_epoch, self.epochs)`"""
        self.epochs = config['epochs']
        r"""End epoch index + 1, aka max iteration times. That is, `epoch_idx` iterates through 
        `range(self.start_epoch, self.epochs)`"""
        self.max_steps = config['max_steps']  # max training batch step

        self.best_valid_timestamp = Timestamp()
        self.valid_intervals = self.config['valid_intervals']
        self.valid_strategy = self.config['valid_strategy']
        self._valid_count = 0
        self.train_loss_list: List[float] = list()
        self.valid_result_dict: Dict[int, EpochTracker] = dict()
        self.stopping_steps = config['stopping_steps']
        self.stopped = False
        self.stopping_count = 0

        # Evaluation strategy
        self.metrics_for_best_model = set(self.config["metrics_for_best_model"])
        self.evaluator = BaseEvaluator(config, self.config["metrics"])

        # Functionality
        ensure_dir(config['checkpoint_dir'])
        self.saved_model_filename: str = os.path.join(config['checkpoint_dir'], self.filename)
        r"""Path to saved checkpoint file (without suffix!), which can be loaded with `load_experiment`."""

        ensure_dir(config['generated_text_dir'])
        self.saved_text_filename: str = os.path.join(config['generated_text_dir'], self.filename)

        self.max_save = config['max_save']
        if self.quick_test and self.max_save is None:
            self.max_save = 0
        self.max_save = self.max_save or 2
        self.disable_tqdm = config['disable_tqdm'] or not self.accelerator.is_local_main_process
        self._summary_tracker = get_dashboard()

    def _build_optimizer(self, optimizer: str, scheduler: Optional[str])\
            -> Union[optim.Optimizer, AbstractScheduler]:
        """Init the optimizer and scheduler.

        Returns:
            Union[optim.Optimizer, AbstractScheduler]: the optimizer
        """

        optimizer_class = collections.defaultdict(lambda: optim.AdamW, {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop,
            'adafactor': transformers.Adafactor,
        })
        scheduler_class = {
            'inverse': InverseSquareRootScheduler,
            'cosine': CosineScheduler,
            'linear': LinearScheduler,
            'constant': ConstantScheduler,
        }

        # dealing with adafactor
        if optimizer == 'adafactor':
            # using adafactor_kwargs in overall.yaml
            if self.grad_clip is not None:
                self.grad_clip = None
                self.logger.warning("Additional optimizer operations like gradient clipping "
                                    "should not be used alongside Adafactor.")
            self.optimizer_kwargs.update(self.adafactor_kwargs)

        # get optimizer (use default value of pytorch if self.optimizer_kwargs is empty)
        self.logger.debug(f'Using optimizer {optimizer}')
        optimizer = optimizer_class[optimizer](
            params=self._trainable_parameters,
            **self.optimizer_kwargs
        )

        # scheduling
        if scheduler is not None and scheduler in scheduler_class:
            assert isinstance(self.scheduler_kwargs, dict), "Please specify scheduler_kwargs"
            self.logger.debug(f'Using scheduler {scheduler}.')
            self.scheduler_kwargs.setdefault("max_lr", self.learning_rate)
            optimizer = scheduler_class[scheduler](
                base_optimizer=optimizer,
                **self.scheduler_kwargs
            )

        return optimizer

    @property
    def timestamp(self) -> Timestamp:
        """Return the timestamp for the moment."""
        return self._summary_tracker.axes

    @property
    def best_valid_result(self) -> EpochTracker:
        """Retrieve best result dict from `self.valid_result_list`."""
        return self.valid_result_dict[self.best_valid_timestamp.valid_epoch]

    def is_save(self) -> bool:
        return self.accelerator.is_local_main_process


    @profile
    def _train_epoch(
            self,
            train_data: DataLoader,
            epoch_idx: int,
            valid_data: Optional[DataLoader] = None,
    ) -> dict:
        r"""Train the model in an epoch

        Args:
            train_data:
            epoch_idx: the current epoch index.
            valid_data: Optional (default = None) the dataloader of validation set

        Returns:
            dict: Training losses.
        """
        self.model.train()
        self._summary_tracker.new_epoch("train")
        if not self.disable_tqdm:
            train_tqdm = tqdm(
                train_data, desc=f"train {epoch_idx:4}", dynamic_ncols=True, postfix={'loss': None}, unit='step'
            )
        else:
            train_tqdm = train_data

        for step, data in enumerate(train_tqdm):
            self._summary_tracker.new_step()
            if self.timestamp.train_step == self.max_steps:
                self.stopped = True
                break

            loss = self.model(data, epoch_idx=epoch_idx)
            # loss = self.accelerator.gather(loss).mean().item()
            self._summary_tracker.append_loss(loss.item())
            self.accelerator.backward(loss / self.accumulation_steps)
            
            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(train_tqdm):
                if self.grad_clip is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if not self.disable_tqdm:
                train_tqdm.set_postfix(loss=self._summary_tracker.epoch_loss)

            if valid_data:
                self.stopped |= self._valid(valid_data, 'step')
            if self.stopped:
                break

        self._summary_tracker.on_epoch_end()

        return self._summary_tracker.epoch_dict()

    @torch.no_grad()
    def _valid(
            self,
            valid_data: DataLoader,
            valid_mode: str,
    ) -> bool:
        """Validate every `self.eval_interval` step or epoch if evaluation strategy matches attribute
        `self.eval_strategy`. Specifically, if `self.eval_interval` is set to `0`, validation will be skipped.

        Early stopping will also be checked if `self.stopping_steps` is positive integer.

        Args:
            valid_data: The dataloader of validation set.
            valid_mode: The evaluation strategy of current call ("epoch" or "step").

        Returns:
            bool: Early stopping. Return true if `self.stopping_steps` is positive integer and `self._early_stopping()`
            is True.
        """
        if (self.valid_intervals <= 0) or (valid_mode != self.valid_strategy):
            return False

        self._valid_count += 1
        if self._valid_count % self.valid_intervals != 0:
            return False

        self._summary_tracker.new_epoch('valid')

        if 'loss' in self.metrics_for_best_model:
            self.model.eval()
            if not self.disable_tqdm:
                valid_tqdm = tqdm(valid_data, desc=f"valid {self.timestamp.valid_epoch:4}", dynamic_ncols=True,
                                  postfix={'loss': None}, unit='step')
            else:
                valid_tqdm = valid_data
            for data in valid_tqdm:
                self._summary_tracker.new_step()
                loss = self.model(data)
                losses = self.accelerator.gather(loss)
                loss = losses.mean()
                self._summary_tracker.append_loss(loss)
                if not self.disable_tqdm:
                    valid_tqdm.set_postfix(loss=self._summary_tracker.epoch_loss)
        else:
            valid_results = self.evaluate(valid_data, is_valid=True, valid_count=self.timestamp.valid_epoch)
            self._summary_tracker.set_metrics_results(valid_results)
        
        self.model.train()

        self.valid_result_dict[self._summary_tracker.axes.valid_epoch] = self._summary_tracker.current_epoch

        self.best_valid_timestamp, current_best = self._summary_tracker.get_best_valid()
        stopped = bool(self.stopping_steps) and self._early_stopping(current_best)

        if self.is_save():
            self.save_checkpoint()
        self.accelerator.wait_for_everyone()

        self._summary_tracker.on_epoch_end()

        return stopped

    def _early_stopping(self, current_best: bool) -> bool:
        r""" Check early stopping with `stopping_steps`, a maximum amount of non-best validation.

        Args:
            current_best: Whether current epoch is the one with the best score.

        Return:
            bool: If true, the training process will be stopped, else the `self.stopping_count` will accumulate.
        """

        stop_flag = False

        if current_best:
            self.stopping_count = 0
        else:
            self.stopping_count += 1
            stop_flag = self.stopping_count > self.stopping_steps

        return stop_flag

    def _get_checkpoint(self) -> Optional[dict]:
        if len(self.valid_result_dict) == 0:
            self.logger.warning('Get checkpoint failed. No validation has been performed.')
            return None

        # construct state_dict and parameters
        _state_dict = self.accelerator.unwrap_model(self.model).state_dict()

        # get optimizer, config and validation summary
        checkpoint = {
            # parameters that needed to be loaded
            'state_dict': _state_dict,
            'optimizer': self.optimizer.state_dict(),
            'stopping_count': self.stopping_count,
            'best_valid_score': self._summary_tracker.best_valid_score,
            'epoch': self.timestamp.train_epoch,
            'timestamp': self.timestamp,
            'config': self.config,
            # parameters for recording only
            'summary': self.valid_result_dict[self._summary_tracker.axes.valid_epoch],
        }
        self.logger.debug(checkpoint)
        return checkpoint

    def save_checkpoint(self):
        serialized_save(
            self._get_checkpoint(),
            serial=self.timestamp.train_epoch,
            serial_of_soft_link=self.best_valid_timestamp.train_epoch,
            path_without_extension=self.saved_model_filename,
            tag='epoch',
            extension_name='pth',
            max_save=self.max_save,
        )

    def save_generated_text(self, generated_corpus: List[str], is_valid: bool = False):
        r"""Store the generated text by our model into `self.saved_text_filename`."""
        if is_valid:
            self._summary_tracker.add_corpus('valid-' + str(self.timestamp.valid_epoch), generated_corpus)
        else:
            self._summary_tracker.add_corpus('test', generated_corpus)
            serialized_save(
                generated_corpus,
                serial=None,
                serial_of_soft_link=None,
                path_without_extension=self.saved_text_filename,
                tag=None,
                extension_name='txt',
            )

    def resume_checkpoint(self, resume_file: str):
        r"""Load the model parameters information and training information.

        Args:
            resume_file: the checkpoint file (specific by `load_experiment`).
        """
        # check
        self.logger.info("Resuming checkpoint from {}...".format(resume_file))
        if os.path.isfile(resume_file):
            checkpoint = torch.load(resume_file, map_location=self.device)
        else:
            self.logger.warning('Checkpoint file "{}" not found. Resuming stopped.'.format(resume_file))
            return

        # load start epoch and early stopping
        self.start_epoch = checkpoint['epoch'] + 1  # start from the next step
        self._summary_tracker.axes = checkpoint['timestamp']
        self.stopping_count = checkpoint['stopping_count']
        self._summary_tracker.best_valid_score = checkpoint['best_valid_score']
        self.valid_result_dict = checkpoint['summary']

        if checkpoint['config']['seed']:
            init_seed(checkpoint['config']['seed'], checkpoint['config']['reproducibility'])
            set_seed(checkpoint['config']['seed'])

        # load architecture params from checkpoint
        if checkpoint['config']['model_name'] != self.config['model_name']:
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        if checkpoint['config']['optimizer'].lower() != self.config['optimizer']:
            self.logger.warning(
                'Optimizer configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

    def fit(
            self,
            train_data: DataLoader,
            valid_data: Optional[DataLoader] = None,
    ) -> dict:
        r"""Train the model based on the train data.

        Args:
            train_data: The dataloader of training set.
            valid_data: (default = None) The dataloader of training set.

        Returns:
             dict: the best valid score and best valid result.
        """

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.logger.info("====== Start training ======")
        self.accelerator.wait_for_everyone()
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            self._train_epoch(train_data, epoch_idx, valid_data)
            self.train_loss_list.append(self._summary_tracker.epoch_loss)

            # valid
            if valid_data:
                self.stopped |= self._valid(valid_data, 'epoch')

            if self.stopped:
                if self.stopping_steps:
                    self.logger.info(f'Early stopped at {self.stopping_count} non-best validation.')
                elif self.max_steps:
                    self.logger.info(f'Stopped at max_steps {self.max_steps}.')
                break

        file = self.saved_model_filename+".pth"
        if os.path.exists(file):
            self.logger.info(f'Soft link created: {file} -> {os.readlink(file)}')
        self.logger.info(f'====== Finished training, best validation result '
                         f'at train epoch {self.best_valid_timestamp.train_epoch} ======')

        self.model = self.accelerator.unwrap_model(self.model)
        self.logger.info('Best valid result: {}'.format(self.best_valid_result.metrics_info()))
        return self.best_valid_result.as_dict()

    @torch.no_grad()
    def evaluate(
            self,
            eval_data: DataLoader,
            load_best_model: bool = True,
            model_file: Optional[str] = None,
            _eval: bool = True,
            is_valid: bool = False,
            valid_count: Optional[int] = None,
    ) -> Optional[dict]:
        r"""Evaluate the model based on the `eval_data`.

        Args:
            eval_data (AbstractDataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            _eval: (default = True) True to evaluate and False to preview generation only.
            is_valid: (default = False) True if evaluate during validation
            valid_count: (default = None) Validation index if `is_valid` is True.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if is_valid:
            load_best_model = False

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_filename + '.pth'
            if not os.path.isfile(checkpoint_file):
                self.logger.error(f'Failed to evaluate model: "{checkpoint_file}" not found. '
                                  f'(You may specify it with `load_experiment`)')
                return None
            self.logger.info('Loading model structure and parameters from {} ...'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.accelerator.wait_for_everyone()
        
        if not is_valid:
            self.model = self.accelerator.prepare(self.model)
        
        self.model.eval()

        # generate
        generate_corpus = []
        eval_tqdm = tqdm(eval_data, desc="generating", dynamic_ncols=True) if not self.disable_tqdm else eval_data
        for batch_data in eval_tqdm:
            generated = self.accelerator.unwrap_model(self.model).generate(batch_data, eval_data, self.accelerator)
            generate_corpus.extend(generated)
        if self.is_chinese_task:
            reference_corpus = eval_data.dataset.target_tokens
        else:
            reference_corpus = eval_data.dataset.target_text
        generate_corpus = generate_corpus[:len(reference_corpus)]
        if self.is_save():
            self.save_generated_text(generate_corpus, is_valid)
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)
        et = EpochTracker(self.metrics_for_best_model)
        et.update_metrics(result)
        if not is_valid:
            self.logger.info('Evaluation result:\n{}'.format(et.metrics_info(sep=",\n", indent=" ")))

        return et.as_dict()
