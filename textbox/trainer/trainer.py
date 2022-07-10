import os
import torch
import torch.optim as optim
import math
import collections

from time import time

from torch.nn import Parameter
from tqdm import tqdm
from logging import getLogger

from .scheduler import (
    AbstractScheduler, InverseSquareRootScheduler, CosineScheduler, LinearScheduler, ConstantScheduler
)
import transformers
from ..evaluator import BaseEvaluator, evaluator_list
from ..utils import ensure_dir, get_local_time, init_seed
from textbox.utils.dashboard import SummaryTracker, get_dashboard

from typing import Dict, Optional, Union, Iterable, Collection, Literal, List, Tuple, Iterator, Any
from ..model.abstract_model import AbstractModel
from ..data.abstract_dataloader import AbstractDataLoader
from textbox import Config
from logging import Logger


class AbstractTrainer:
    r"""Trainer Class is used to manage the training and evaluation processes of text generation system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config: Config, model: AbstractModel):
        self.config = config
        self.model = model
        self.logger = getLogger()

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
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.
    """

    def __init__(self, config: Config, model: AbstractModel):
        super(Trainer, self).__init__(config, model)

        self.DDP: bool = config['DDP']
        self.device: torch.device = config['device']
        self.filename = self.config['filename']

        self.stopping_step: Optional[int] = config['stopping_step']
        self.stopped = False
        self.stopping_count = 0

        self.learning_rate: float = config['learning_rate']
        self.weight_decay: float = config['weight_decay']
        self.adam_beta1: Optional[float] = config['adam_beta1']  # default values set in config yaml?
        self.adam_beta2: Optional[float] = config['adam_beta2']
        self.epsilon: Optional[float] = config['epsilon']
        self.max_steps: Optional[int] = config['max_steps']
        self.init_lr: Optional[float] = config['init_lr']
        self.warmup_steps: Optional[int] = config['warmup_steps']
        self.grad_clip: bool = config['grad_clip']
        self._trainable_parameters: Iterator[Parameter] = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = self._build_optimizer(config['optimizer'], config['scheduler'])

        ensure_dir(config['checkpoint_dir'])
        self.saved_model_filename: str = os.path.join(config['checkpoint_dir'], self.filename)
        r"""Path to saved checkpoint file (without suffix!), which can be loaded with `load_experiment`."""

        ensure_dir(config['generated_text_dir'])
        self.saved_text_filename: str = os.path.join(config['generated_text_dir'], self.filename)

        self.start_epoch = 0
        r"""Start epoch index. That is, `epoch_idx` iterates through `range(self.start_epoch, self.epochs)`"""
        self.epochs: int = config['epochs']
        r"""End epoch index + 1, aka max iteration times. That is, `epoch_idx` iterates through 
        `range(self.start_epoch, self.epochs)`"""
        self.epoch_idx: int = -1
        r"""current epoch index"""

        self.best_epoch = -1
        self._best_valid_score = -math.inf
        self.eval_interval, self.eval_strategy = self._set_eval_mode()
        self._eval_count = 0
        self.train_loss_list: List[float] = list()
        self.valid_result_list: List[dict] = list()

        self.metrics: List[str] = self.config["metrics"]
        self.metrics_key: List[str] = self.config["metrics_key"]  # todo: how to access metrics scores? (nested dict)
        self.evaluator = BaseEvaluator(config, self.metrics)

        self.disable_tqdm = self.DDP and torch.distributed.get_rank() != 0
        self.logdir = './log/'
        self._dashboard_getter = get_dashboard(self.logdir, config)
        self._summary_tracker: Optional[SummaryTracker] = None

    def _set_eval_mode(self) -> Tuple[int, Literal["epoch", "step"]]:
        r"""Check evaluation mode. Default = (1, "epoch") (If both `eval_epoch` and `eval_step` are specified,
        `eval_step` is ignored. If both are set to 0, `eval_epoch` is set to 1.)

        Returns:
            Tuple[int, Literal["epoch", "step"]]: a tuple of (evaluation interval, evaluation mode)
        """
        # default value
        eval_epoch = self.config['eval_epoch'] or 0
        eval_step = self.config['eval_step'] or 0

        # check
        if eval_epoch > 0 and eval_step > 0:
            self.logger.warning(
                '"eval_step" and "eval_epoch" are specified at the same time. "eval_step" has been ignored.'
            )
            eval_step = 0
        elif eval_epoch <= 0 and eval_step <= 0:
            self.logger.warning(
                '"eval_step" and "eval_epoch" are set to 0 at the same time. "eval_epoch" has been changed to 1.'
            )
            eval_epoch = 1

        if eval_epoch > 0:
            self.logger.info(f"eval strategy: validate every {eval_epoch} epoch")
            return eval_epoch, "epoch"
        else:
            self.logger.info(f"eval strategy: validate every {eval_step} step")
        return eval_step, "step"

    def _build_optimizer(self, optimizer: Optional[str], scheduler: Optional[str])\
            -> Union[optim.Optimizer, AbstractScheduler]:
        """Init the optimizer and scheduler.

        Returns:
            Union[optim.Optimizer, AbstractScheduler]: the optimizer
        """

        def _get_base_optimizer(name: str) -> optim.Optimizer:

            defaults: Dict[str, Any] = dict(
                params=self._trainable_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

            if name == 'adam':
                _optim = optim.Adam
                defaults.update(betas=(self.adam_beta1, self.adam_beta2), eps=self.epsilon)
            elif name == 'sgd':
                _optim = optim.SGD
            elif name == 'adagrad':
                _optim = optim.Adagrad
                defaults.update(eps=self.epsilon)
            elif name == 'rmsprop':
                _optim = optim.RMSprop
                defaults.update(eps=self.epsilon)
            elif name == 'adamw':
                _optim = optim.AdamW
                defaults.update(betas=(self.adam_beta1, self.adam_beta2), eps=self.epsilon)
            elif name == 'adafactor':
                _optim = transformers.Adafactor
                defaults.update(eps=self.epsilon, beta1=self.adam_beta1)
            else:
                self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
                _optim = optim.Adam
                defaults.update(betas=(self.adam_beta1, self.adam_beta2), eps=self.epsilon)

            return _optim(**defaults)

        def _get_scheduler(name: Optional[str], _optim: optim.Optimizer) -> Union[optim.Optimizer, AbstractScheduler]:

            if name is None:
                return _optim

            defaults = dict(
                base_optimizer=_optim,
                init_lr=self.init_lr,
                max_lr=self.learning_rate,
                n_warmup_steps=self.warmup_steps
            )

            if name == 'inverse':
                _scheduler = InverseSquareRootScheduler
            elif name == 'cosine':
                _scheduler = CosineScheduler
                defaults.update(max_steps=self.max_steps)
            elif name == 'linear':
                _scheduler = LinearScheduler
                defaults.update(max_steps=self.max_steps)
            elif name == 'constant':
                _scheduler = ConstantScheduler
            else:
                self.logger.info("Learning rate scheduling disabled.")
                return _optim

            return _scheduler(**defaults)

        optimizer = _get_base_optimizer(optimizer)
        optimizer = _get_scheduler(scheduler, optimizer)
        return optimizer

    def _train_epoch(
            self,
            train_data: AbstractDataLoader,
            epoch_idx: int,
            valid_data: Optional[AbstractDataLoader] = None,
    ) -> dict:
        r"""Train the model in an epoch

        Args:
            train_data:
            epoch_idx: the current epoch index.
            valid_data: Optional (default = None) the dataloader of validation set

        Returns:
            EpochTracker: :class:`EpochTracker` that includes epoch information.
        """
        self.model.train()
        self._summary_tracker.new_epoch("train")
        if not self.disable_tqdm:
            train_tqdm = tqdm(train_data, desc=f"train {epoch_idx:4}", dynamic_ncols=True, postfix={'loss': None})
        else:
            train_tqdm = train_data

        for data in train_tqdm:
            self.optimizer.zero_grad()

            loss = self.model(data, epoch_idx=epoch_idx)
            self._summary_tracker.append_loss(loss)
            if not self.disable_tqdm:
                train_tqdm.set_postfix(loss=self._summary_tracker.epoch_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if valid_data:
                self.stopped &= self._valid(valid_data, epoch_idx, 'step')
                if self.stopped:
                    break

        self._summary_tracker.on_epoch_end()

        return self._summary_tracker.epoch_dict()

    @torch.no_grad()
    def _valid_epoch(
            self,
            valid_data: AbstractDataLoader,
            idx: int,
    ) -> dict:
        r"""Valid the model with `valid_data`

        Args:
            valid_data: the dataloader of validation set
            idx: the current epoch index.

        Returns:
            EpochTracker: :class:`EpochTracker` that includes epoch information.

        Todo:
            * perform either loss or evaluation according to 'eval_metrics'
        """

        self._summary_tracker.new_epoch('valid')

        if 'loss' in self.metrics:
            self.model.eval()
            if not self.disable_tqdm:
                valid_tqdm = tqdm(valid_data, desc=f"valid {idx:4}", dynamic_ncols=True, postfix={'loss': None})
            else:
                valid_tqdm = valid_data
            for data in valid_tqdm:
                loss = self.model(data)
                self._summary_tracker.append_loss(loss)
                if not self.disable_tqdm:
                    valid_tqdm.set_postfix(loss=self._summary_tracker.epoch_loss)
        else:
            valid_results = self.evaluate(valid_data, load_best_model=False, is_valid=True)
            self._summary_tracker.set_metrics_results(valid_results)

        self._summary_tracker.on_epoch_end()

        return self._summary_tracker.epoch_dict()

    def _valid(
            self,
            valid_data: AbstractDataLoader,
            epoch_idx: int,
            eval_strategy: Literal["epoch", "step"],
    ) -> bool:
        """Validate every `self.eval_interval` step or epoch if evaluation strategy matches attribute
        `self.eval_strategy`. Specifically, if `self.eval_interval` is set to `0`, validation will be skipped.

        Early stopping will also be checked if `self.stopping_step` is positive integer.

        Args:
            valid_data: The dataloader of validation set.
            epoch_idx: The current epoch index.
            eval_strategy: The evaluation strategy of current call ("epoch" or "step").

        Returns:
            bool: Early stopping. Return true if `self.stopping_step` is positive integer and `self._early_stopping()`
            is True.

        Todo:
            * Update docstring: valid method 'loss' or 'metrics'
        """
        if (eval_strategy != self.eval_strategy) or (self.eval_interval <= 0):
            return False

        self._eval_count += 1
        if self._eval_count % self.eval_interval != 0:
            return False

        self._valid_epoch(valid_data, self._eval_count // self.eval_interval)
        self.valid_result_list.append(self._summary_tracker.epoch_dict())

        stopped = bool(self.stopping_step) and self._early_stopping(self._summary_tracker.epoch_score, epoch_idx)
        self.save_checkpoint()

        return stopped

    @property
    def best_result(self) -> dict:
        return self.valid_result_list[self.best_epoch]

    def save_checkpoint(self, tag: Optional[str] = None, overwrite: bool = True):
        r"""Store the model parameters information and training information.

        Save checkpoint every validation as the formate of 'Model-Dataset-Time_epoch-?.pth'. A soft link named
        'Model-Dataset-Time.pth' pointing to best epoch will be created.

        Todo:
            * Checkpoint save which files?
        """
        if len(self.valid_result_list) == 0:
            self.logger.warning('Save checkpoint failed. No validation has been performed.')
            return

        # construct state_dict and parameters
        _state_dict = self.model.state_dict()
        if self.DDP and torch.distributed.get_rank() == 0:
            _new_dict = dict()
            for key, val in _state_dict["state_dict"].items():
                changed_key = key[7:] if key.startswith('module.') else key
                _new_dict[changed_key] = val
            _state_dict = _new_dict

        # get optimizer, config and validation summary
        checkpoint = {
            'state_dict': _state_dict,
            'optimizer': self.optimizer.state_dict(),
            'stopping_count': self.stopping_count,
            'best_valid_score': self._best_valid_score,
            'epoch': self.epoch_idx,
            'config': self.config,
            'summary': self.valid_result_list[-1],  #todo add text
        }

        self.logger.debug(checkpoint)

        # deal with naming
        if tag is None:
            tag = '_epoch-' + str(self.epoch_idx)
        path_to_save = os.path.abspath(self.saved_model_filename + tag)
        if os.path.exists(path_to_save):
            if overwrite:
                os.remove(path_to_save)  # behavior of torch.save is not clearly defined.
            else:
                path_to_save += get_local_time()
        path_to_save += '.pth'

        torch.save(checkpoint, path_to_save)
        self.logger.info('Saving current: {}'.format(path_to_save))

        # create soft link to best model
        if self.best_epoch == self.epoch_idx:
            path_to_best = os.path.abspath(self.saved_model_filename + '.pth')
            if os.path.exists(path_to_best):
                os.remove(path_to_best)
            os.symlink(path_to_save, path_to_best)

    def _save_generated_text(self, generated_corpus: List[str]):
        r"""Store the generated text by our model into `self.saved_text_filename`."""
        with open(self.saved_text_filename + '.txt', 'w') as fout:
            for text in generated_corpus:
                fout.write(text + '\n')

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
        self.start_epoch = checkpoint['epoch'] + 1
        self.stopping_count = checkpoint['stopping_count'] or checkpoint['cur_step']
        self._best_valid_score = checkpoint['_best_valid_score']

        if checkpoint['config']['seed']:
            init_seed(checkpoint['config']['seed'], checkpoint['config']['reproducibility'])

        # load architecture params from checkpoint
        if checkpoint['config']['model_name'] != self.config['model_name']:
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        if self.DDP:
            saved_dict = collections.OrderedDict()
            for state_dict_key, state_dict_val in checkpoint['state_dict'].items():
                changed_key = 'module.' + state_dict_key
                saved_dict[changed_key] = state_dict_val
            checkpoint['state_dict'] = saved_dict
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
            train_data: AbstractDataLoader,
            valid_data: Optional[AbstractDataLoader] = None,
    ) -> dict:
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data: The dataloader of training set.
            valid_data: (default = None) The dataloader of training set.

        Returns:
             dict: the best valid score and best valid result.

        Todo:
            * Complete the docstring.
            * Modify the return value.
        """
        self.logger.info("====== Start training ======")
        with self._dashboard_getter() as summary_tracker:
            self._summary_tracker = summary_tracker
            for epoch_idx in range(self.start_epoch, self.epochs):
                self.epoch_idx = epoch_idx
                # train
                self._train_epoch(train_data, epoch_idx, valid_data)
                self.train_loss_list.append(summary_tracker.epoch_loss)

                # valid
                if valid_data:
                    self.stopped &= self._valid(valid_data, epoch_idx, 'epoch')
                    if self.stopped:
                        break

        self.logger.info('====== Finished training, best eval result in epoch {} ======'.format(self.best_epoch))

        return self.best_result

    def _early_stopping(self, score: float, epoch_idx: int) -> bool:
        r""" Return True if valid score has been lower than best score for `stopping_step` steps.

        Args:
            valid_tracker: contain valid information (loss and metrics score)

        Todo:
            * Abstract results.
        """

        stop_flag = False

        if score > self._best_valid_score:
            self._best_valid_score = score
            self.stopping_count = 0
            self.best_epoch = epoch_idx  # get best results with index
        else:
            self.stopping_count += 1
            stop_flag = self.stopping_count > self.stopping_step

        return stop_flag

    @torch.no_grad()
    def evaluate(
            self,
            eval_data: AbstractDataLoader,
            load_best_model: bool = True,
            model_file: Optional[str] = None,
            _eval: bool = True,
            is_valid: bool = False,
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

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value

        Todo:
            * perform either loss or evaluation according to 'eval_metrics'
        """
        if is_valid and load_best_model:
            load_best_model = False
            self.logger.warning('Evaluation should not load best model during validation. `load_best_model` has'
                                'set to False temporarily.')

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_filename + '.pth'
            if not os.path.isfile(checkpoint_file):
                self.logger.error(f'Failed to evaluate model: "{checkpoint_file}" not found.')
                return None
            self.logger.info('Loading model structure and parameters from {} ...'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()

        # preview only
        if not _eval or len(self.metrics) == 0:
            generate_sentence = self.model.generate(next(eval_data), eval_data)
            generate_sentence = ' '.join(generate_sentence[0])
            self.logger.info('Generation Preview: ' + generate_sentence)
            return {"preview": generate_sentence}

        # generate
        generate_corpus = []
        eval_tqdm = tqdm(eval_data, desc="generating", dynamic_ncols=True) if not self.disable_tqdm else eval_data
        for batch_data in eval_tqdm:
            generated = self.model.generate(batch_data, eval_data)
            generate_corpus.extend(generated)
        if not is_valid:
            self._save_generated_text(generate_corpus)
        if not self._summary_tracker.tracker_finished:
            self._summary_tracker.add_corpus('corpus', generate_corpus)
        reference_corpus = eval_data.dataset.target_text
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)

        return result
