import os
import torch
import torch.optim as optim
import numpy as np
import math
import collections

from tqdm import tqdm

from logging import getLogger

from textbox.module.scheduler import (
    AbstractScheduler, InverseSquareRootScheduler, CosineScheduler, LinearScheduler, ConstantScheduler
)
from textbox.evaluator import BaseEvaluator, evaluator_list
from textbox.utils.utils import ensure_dir, Timer, ordinal
from textbox.utils.dashboard import get_dashboard, AbstractDashboard, TensorboardWriter, NilWriter

from typing import Dict, Optional, Union, Set, Collection, Literal, List, Tuple, Iterable
from textbox.model.abstract_model import AbstractModel
from textbox.data.abstract_dataloader import AbstractDataLoader
from textbox.config.configurator import Config
from logging import Logger
MetricsDict = Dict[str, Union[float, Dict[str, float], Collection[float]]]


class EpochTracker:
    r"""
    Track and visualize validating metrics results and training / validating loss.
    Dashboard now supports :class:`textbox.utils.TensorboardWriter`.

    Args:
        epoch_idx: Current epoch index.
        DDP: Whether the Pytorch DDP is enabled.
        train: Optional (default = True)
            Train or eval mode.
        dashboard: Optional (default = None)
            Implementation of visualization toolkit.

    Example:
        >>> tbw = TensorboardWriter("log/dir")
        >>> for epoch in range(10):
        >>>     valid_tracker = EpochTracker(epoch, DDP=False, train=False, dashboard=tbw)
        >>>     for step in range(10):
        >>>         valid_tracker.append_loss(1.0)
        >>>     valid_tracker.set_result({"metric1": 1.0})
        >>>     valid_tracker.info(time_duration=10.0)
        Epoch 0,  validating [time: 10.00s, validating_loss: 1.0000, ppl: 2.71]
    """
    _is_train: Dict[bool, Literal["training", "validating"]] = {True: "training", False: "validating"}

    def __init__(
            self,
            epoch_idx: int,
            DDP: bool,
            train: bool = True,
            dashboard: Optional[AbstractDashboard] = None
    ):
        self.epoch_idx: int = epoch_idx

        self._accumulate_loss: float = 0.
        self._accumulate_step: int = 0
        self._loss: Optional[float] = None

        self._metrics_results: Optional[MetricsDict] = None
        self._score: Optional[float] = None

        self._logger: Logger = getLogger()
        self._DDP: bool = DDP
        self.mode_tag = self._is_train[train]

        self._dashboard: AbstractDashboard = dashboard or NilWriter()

    def append_loss(self, loss: float):
        r"""Append loss of current step to tracker and update current step."""
        _check_nan(loss)
        self._dashboard.add_scalar("Loss/" + self.mode_tag, loss)
        self._dashboard.step()

        self._accumulate_loss += loss
        self._accumulate_step += 1

    def set_result(self, results: dict):
        r"""Record the metrics results."""
        for metric, result in results.items():
            self._dashboard.add_any("Metrics/" + metric, result)
        self._metrics_results = results

    def _calc_score(self) -> Optional[float]:
        r"""Ensure the score will only be calculated and recorded once."""
        if self._metrics_results is None:
            return None

        score = 0.
        for metric, result in self._metrics_results.items():
            if isinstance(result, dict):
                float_list = list(filter(lambda x: isinstance(x, float), result.values()))
            elif isinstance(result, Collection):
                float_list = list(filter(lambda x: isinstance(x, float), result))
            elif isinstance(result, float):
                float_list = [result]
            else:
                self._logger.warning(f"Failed when working out score of metric {metric}.")
                continue
            if len(float_list) != 0:
                score += sum(float_list) / len(float_list)
        self._dashboard.add_scalar("Metrics/score", score)

        return score

    @property
    def score(self) -> Optional[float]:
        r"""Get the total sum score of metric results in evaluating epochs."""
        if self.mode_tag == self._is_train[True]:
            self._logger.warning('`score` is unavailable training epochs.')
            return None

        if not self._score:
            self._score = self._calc_score()
        return self._score

    def _calc_loss(self) -> float:
        r"""Ensure the loss will only be calculated and reduced once if DDP is enabled."""
        if self._accumulate_step == 0:
            self._logger.warning("Trying to access epoch average loss before append any.")
            return math.inf
        _loss = self._accumulate_loss / self._accumulate_step
        if self._DDP:
            _loss = torch.tensor(_loss).to("cuda")
            torch.distributed.all_reduce(_loss, op=torch.distributed.ReduceOp.SUM)
            _loss /= torch.distributed.get_world_size()
            _loss = _loss.item()
        return _loss

    @property
    def loss(self) -> float:
        r"""Get total loss (loss will be reduced on the first call)."""
        if not self._loss:
            self._loss = self._calc_loss()
        return self._loss

    @property
    def perplexity(self) -> float:
        r"""Get exponent of total loss."""
        return np.exp(self.loss)

    def info(self, time_duration: float, extra_info: str = ""):
        r"""Output loss with epoch and time information."""
        output = f"Epoch {self.epoch_idx}, {extra_info} {self.mode_tag} "\
                 f"[time: {time_duration:.2f}s, {self.mode_tag}_loss: {self.loss:.4f}"

        if self.mode_tag == "validating":
            output += f", ppl: {self.perplexity:.4f}"
            for metric, result in self._metrics_results.items():
                if metric != 'loss':
                    output += f", {metric}: {result}"
        output += "]"

        # flush
        self._logger.info(output)

    def __getitem__(self, item: str) -> Optional[float]:
        if self._metrics_results is None:
            self._logger.warning("Please set result first.")
            return None
        return self._metrics_results[item]


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
        self.embedding_size: int = config['embedding_size']
        self.filename = self.config['filename']

        self.stopping_step: Optional[int] = config['stopping_step']
        self.stopped = False
        self.stopping_count = 0
        self.init_lr: Optional[float] = config['init_lr']
        self.warmup_steps: Optional[int] = config['warmup_steps']
        self.max_steps: Optional[int] = config['max_steps']
        self.learning_rate: float = config['learning_rate']
        self.grad_clip: bool = config['grad_clip']
        self.optimizer = self._build_optimizer(config['optimizer'], config['scheduler'])

        ensure_dir(config['checkpoint_dir'])
        _saved_model_file: str = self.filename + '.pth'
        self.saved_model_file: str = os.path.join(config['checkpoint_dir'], _saved_model_file)
        r"""Path to saved checkpoint file, which can be loaded with `load_experiment`."""

        ensure_dir(config['generated_text_dir'])
        _saved_text_file: str = self.filename + '.txt'
        self.saved_text_file: str = os.path.join(config['generated_text_dir'], _saved_text_file)

        self.start_epoch = 0
        r"""Start epoch index. That is, `epoch_idx` iterates through `range(self.start_epoch, self.epochs)`"""
        self.epochs: int = config['epochs']
        r"""End epoch index + 1, aka max iteration times. That is, `epoch_idx` iterates through 
        `range(self.start_epoch, self.epochs)`"""
        self.epoch_idx: int = -1
        r"""current epoch index"""

        self.best_epoch = -1
        self.best_valid_score = -math.inf
        self.best_valid_loss = math.inf
        self.best_valid_ppl = math.inf
        self._saved_once = False
        self.eval_interval, self.eval_mode = self._set_eval_mode()
        self._eval_count = 0
        self.test_batch_size: int = config['eval_batch_size']
        self.train_loss_list = list()

        self.metrics = self._process_metrics("metrics")
        self.evaluator = BaseEvaluator(config, self.metrics)
        self.valid_metrics = self._process_metrics("valid_metrics")
        self.valid_evaluator = BaseEvaluator(config, self.valid_metrics)

        self._is_logger = (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP
        self.item_tensor = None
        self.tot_item_num = None
        self.iid_field = config['ITEM_ID_FIELD']
        self._DashboardClass = get_dashboard(config['dashboard'])
        self._dashboard: Optional[AbstractDashboard] = None
        self.logdir = './log/'

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
                '"eval_step" and "eval_epoch" are set to 0 at the same time. "eval_epoch" has been set to 1.'
            )
            eval_epoch = 1

        if eval_epoch > 0:
            self.logger.info(f"eval mode: validate every {eval_epoch} epoch")
            return eval_epoch, "epoch"
        else:
            self.logger.info(f"eval mode: validate every {eval_step} step")
        return eval_step, "step"

    def _build_optimizer(self, optimizer: Optional[str], scheduler: Optional[str])\
            -> Union[optim.Optimizer, AbstractScheduler]:
        """Init the optimizer and scheduler.

        Returns:
            Union[optim.Optimizer, AbstractScheduler]: the optimizer
        """

        def _get_base_optimizer(name: str) -> optim.Optimizer:

            name = name.lower()

            if name == 'adam':
                _optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif name == 'sgd':
                _optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)
            elif name == 'adagrad':
                _optim = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
            elif name == 'rmsprop':
                _optim = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
            else:
                self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
                _optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            return _optim

        def _get_scheduler(name: Optional[str], _optim: optim.Optimizer) -> Union[optim.Optimizer, AbstractScheduler]:

            if name is None:
                return _optim

            name = name.lower()

            assert isinstance(self.init_lr, float), "Specify initial learning rate (`init_lr`)"
            assert isinstance(self.warmup_steps, int), "Specify warmup steps (`warmup_steps`)"

            if name == 'inverse':
                _optim = InverseSquareRootScheduler(_optim, self.init_lr, self.learning_rate, self.warmup_steps)
            elif name == 'cosine':
                assert isinstance(self.max_steps, int), "Specify max steps (`max_steps`)"
                _optim = CosineScheduler(_optim, self.init_lr, self.learning_rate, self.warmup_steps, self.max_steps)
            elif name == 'linear':
                assert isinstance(self.max_steps, int), "Specify max steps (`max_steps`)"
                _optim = LinearScheduler(_optim, self.init_lr, self.learning_rate, self.warmup_steps, self.max_steps)
            elif name == 'constant':
                _optim = ConstantScheduler(_optim, self.init_lr, self.learning_rate, self.warmup_steps)
            else:
                self.logger.info("Learning rate scheduling disabled.")

            return _optim

        optimizer = _get_base_optimizer(optimizer)
        optimizer = _get_scheduler(scheduler, optimizer)
        return optimizer

    def _process_metrics(self, metrics_type: Literal["metrics", "valid_metrics"]) -> Set[str]:
        r"""Get and check the user-specific metrics. The function get metrics from config, checks if it's a string or
        a list, and then checks if the metrics are in the supported list.

        Args:
            metrics_type: "metrics" or "valid_metrics"

        Returns:
            Set[str]: A set of metrics. An empty set will be returned if none metric is specified.

        Raises:
            TypeError: If the `self.config[metrics_type]` is not a string or a list.
        """
        # Type check and pre-process
        metrics = self.config[metrics_type]
        if not metrics:
            return set()
        if not isinstance(metrics, (str, list)):
            raise TypeError(f'Evaluator(s) of {metrics_type} must be a string or list, not {type(metrics)}')
        if isinstance(metrics, str):
            if metrics[0] == '[' and metrics[-1] == ']':
                metrics = metrics[1:-1]
            metrics = metrics.split(",")
        metrics = set(map(lambda x: x.strip().lower(), metrics))

        # Implementation Check
        if len(metrics - evaluator_list) != 0:
            self.logger.warning(
                f"Evaluator(s) of {metrics_type} " + ", ".join(metrics - evaluator_list) +
                " are ignored because are not in supported evaluators list (" + ", ".join(evaluator_list) + ")."
            )
            metrics -= metrics - evaluator_list

        return metrics

    def _train_epoch(
            self,
            train_data: AbstractDataLoader,
            epoch_idx: int,
            valid_data: Optional[AbstractDataLoader] = None,
            process_bar: bool = True,
    ) -> EpochTracker:
        r"""Train the model in an epoch

        Args:
            train_data:
            epoch_idx: the current epoch index.
            valid_data: Optional (default = None) the dataloader of validation set
            process_bar: Optional (default = True) True to show process bar.

        Returns:
            EpochTracker: :class:`EpochTracker` that includes epoch information.
        """
        self.model.train()
        tracker = EpochTracker(epoch_idx, self.DDP, train=True, dashboard=self._dashboard)
        train_tqdm = tqdm(train_data, desc=f"train {epoch_idx:4}", ncols=80) if process_bar and self._is_logger \
            else train_data

        for data in train_tqdm:
            self.optimizer.zero_grad()

            loss = self.model(data, epoch_idx=epoch_idx)
            tracker.append_loss(loss)

            loss.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if valid_data:
                self.stopped &= self._valid(valid_data, epoch_idx, 'step')
                if self.stopped:
                    break

        return tracker

    def _valid_epoch(
            self,
            valid_data: AbstractDataLoader,
            epoch_idx: int,
            process_bar: bool = True,
    ) -> EpochTracker:
        r"""Valid the model with `valid_data`

        Args:
            valid_data: the dataloader of validation set
            epoch_idx: the current epoch index.

        Returns:
            EpochTracker: :class:`EpochTracker` that includes epoch information.
        """
        self.model.eval()
        tracker = EpochTracker(epoch_idx, self.DDP, train=False, dashboard=self._dashboard)
        valid_tqdm= tqdm(valid_data, desc=f"train {epoch_idx:4}", ncols=80) if process_bar and self._is_logger \
            else valid_data

        for data in valid_tqdm:
            losses = self.model(data)
            tracker.append_loss(losses)

        valid_results = self.evaluate(valid_data, load_best_model=False, is_valid=True)
        tracker.set_result(valid_results)

        return tracker

    def _valid(
            self,
            valid_data: AbstractDataLoader,
            epoch_idx: int,
            position: Literal["epoch", "step"],
    ) -> bool:
        """Validate every `self.eval_interval` step or epoch if invoke position matches attribute `self.eval_mode`.
        Specifically, if `self.eval_interval` is set to `0`, validation will be skipped.

        Early stopping will also be checked if `self.stopping_step` is positive integer.

        Args:
            valid_data: The dataloader of validation set.
            epoch_idx: The current epoch index.
            position: Invoke position of method ("epoch" or "step").

        Returns:
            bool: Early stopping. Return true if `self.stopping_step` is positive integer and `self._early_stopping()`
            is True.
        """
        if (position != self.eval_mode) or (self.eval_interval <= 0):
            return False

        self._eval_count += 1
        if self._eval_count % self.eval_interval != 0:
            return False

        with torch.no_grad(), Timer() as valid_timer:
            valid_tracker = self._valid_epoch(valid_data, epoch_idx)
        valid_tracker.info(valid_timer.duration, extra_info=ordinal(self._eval_count // self.eval_interval))

        stopped = bool(self.stopping_step) and self._early_stopping(valid_tracker, epoch_idx)

        return stopped

    def _save_checkpoint(self):
        r"""Store the model parameters information into `self.saved_model_file` and training information.

        Todo:
            * Update checkpoint format
        """
        if not self.saved_model_file:
            return

        _state_dict = self.model.state_dict()
        if self.DDP and torch.distributed.get_rank() == 0:
            _new_dict = collections.OrderedDict()
            for key, val in _state_dict["state_dict"].items():
                changed_key = key[7:] if key[0:7] == 'module.' else key
                _new_dict[changed_key] = val
            _state_dict = _new_dict

        state = {
            'config': self.config,
            'epoch': self.epoch_idx,
            'current': self.stopping_count,
            'best_valid_score': self.best_valid_score,
            'state_dict': _state_dict,
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, self.saved_model_file)
        self._saved_once = True
        self.logger.info('Saving current: {}'.format(self.saved_model_file))

    def _save_generated_text(self, generated_corpus: List[List[str]]):
        r"""Store the generated text by our model into `self.saved_text_file`."""
        with open(self.saved_text_file, 'w') as fout:
            for tokens in generated_corpus:
                fout.write(' '.join(tokens) + '\n')

    def resume_checkpoint(self, resume_file: str):
        r"""Load the model parameters information and training information.

        Args:
            resume_file: the checkpoint file (specific by `load_experiment`).

        Todo:
            * Update checkpoint format
        """
        self.logger.info("Resuming checkpoint from {}...".format(resume_file))
        if os.path.isfile(resume_file):
            checkpoint = torch.load(resume_file, map_location=self.device)
        else:
            self.logger.warning('Checkpoint file "{}" not found. Resuming stopped.'.format(resume_file))
            return

        self.start_epoch = checkpoint['epoch'] + 1
        self.stopping_count = checkpoint['stopping_count'] or checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
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
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

    def fit(
            self,
            train_data: AbstractDataLoader,
            valid_data: Optional[AbstractDataLoader] = None,
            saved: bool = True
    ) -> dict:
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data: The dataloader of training set.
            valid_data: (default = None) The dataloader of training set.
            saved: (default = True) True if checkpoints of best epochs are about to be saved.

        Returns:
             dict: the best valid score and best valid result.

        Todo:
            * Complete the docstring.
            * Modify the return value.
        """
        self.logger.info("====== Start training ======")
        with self._DashboardClass(self.logdir, self.filename) as dashboard:
            self._dashboard = dashboard
            for epoch_idx in range(self.start_epoch, self.epochs):
                self.epoch_idx = epoch_idx
                # train
                with Timer() as train_timer:
                    train_tracker = self._train_epoch(train_data, epoch_idx, valid_data)
                self.train_loss_list.append(train_tracker.loss)
                train_tracker.info(train_timer.duration)

                # valid
                if valid_data:
                    self.stopped &= self._valid(valid_data, epoch_idx, 'epoch')
                    if self.stopped:
                        break

        if not self._saved_once and saved:
            self._save_checkpoint()
        self.logger.info('====== Finished training, best eval result in epoch {} ======'.format(self.best_epoch))

        result = {"best_valid_score": self.best_valid_score,
                  "best_valid_loss": self.best_valid_loss,
                  "best_valid_ppl": self.best_valid_ppl}
        return result

    def _early_stopping(self, valid_tracker: EpochTracker, epoch_idx: int) -> bool:
        r""" Return True if valid score has been lower than best score for `stopping_step` steps.

        Args:
            valid_tracker: contain valid information (loss, perplexity and metrics score)

        Todo:
            * Abstract results.
        """

        stop_flag = False

        if valid_tracker.score > self.best_valid_score:
            self.stopping_count = 0
            self.best_valid_score = valid_tracker.score
            self.best_valid_ppl = valid_tracker.perplexity
            self.best_epoch = epoch_idx
            self._save_checkpoint()
        else:
            self.stopping_count += 1
            stop_flag = self.stopping_count > self.stopping_step

        return stop_flag

    def _evaluate_nll_test(self, eval_data: AbstractDataLoader) -> float:
        r"""Calculate the negative log-likelihood of the eval_data.

        Args:
            eval_data (AbstractDataLoader): the eval data.

        Returns:
            float: NLL_test of the eval data.
        """

        total_loss = 0
        for epoch_idx, eval_batch in enumerate(eval_data):
            nll_test = self.model.calculate_nll_test(eval_batch, epoch_idx)
            total_loss += float(nll_test)
        return total_loss / len(eval_data)

    @torch.no_grad()
    def evaluate(
            self,
            eval_data: AbstractDataLoader,
            load_best_model: bool = True,
            model_file: Optional[str] = None,
            _eval: bool = True,
            is_valid: bool = False
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
        """
        if is_valid:
            _evaluator = self.valid_evaluator
            _metrics = self.valid_metrics
            if load_best_model:
                load_best_model = False
                self.logger.warning('Evaluation should not load best model during validation. `load_best_model` has'
                                    'set to False temporarily.')
        else:
            _evaluator = self.evaluator
            _metrics = self.metrics

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            if not os.path.isfile(checkpoint_file):
                self.logger.error(f'Failed to evaluate model: "{checkpoint_file}" not found.')
                return None
            self.logger.info('Loading model structure and parameters from {} ...'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()

        # preview only
        if not _eval or len(_metrics) == 0:
            generate_sentence = self.model.generate(next(eval_data), eval_data)
            generate_sentence = ' '.join(generate_sentence[0])
            self.logger.info('Generation Preview: ' + generate_sentence)
            return {"preview": generate_sentence}

        # generate
        generate_corpus = []
        eval_tqdm = tqdm(eval_data, desc="generating", ncols=80) if self._is_logger else eval_data
        for batch_data in eval_tqdm:
            generated = self.model.generate(batch_data, eval_data)
            assert len(generated) == len(batch_data['target_text']), "Generated corpus has a mismatched batch size!"
            generate_corpus.extend(generated)
        self._save_generated_text(generate_corpus)

        # evaluation
        reference_corpus = eval_data.get_reference()
        result = _evaluator.evaluate(generate_corpus, reference_corpus)
        if "nll_test" in _metrics and self.config['task_type'].lower() == "unconditional":
            result['nll_test'] = self._evaluate_nll_test(eval_data)

        return result


def _check_nan(value: Union[torch.Tensor, float]):
    r"""Not-a-number check

    Raises:
        ValueError: If `value` is nan.
    """
    if (isinstance(value, torch.Tensor) and torch.isnan(value)) \
            or (isinstance(value, float) and math.isnan(value)):
        raise ValueError('Value is nan.')
