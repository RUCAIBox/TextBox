"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`
"""
from contextlib import contextmanager
from copy import copy
from logging import getLogger
import math
import os
from time import time
from typing import Dict, Iterable, Optional, Set, Union, List, Callable, Tuple
import traceback

import pandas as pd
from pandas.core.resample import f
import torch
import wandb
from wandb import AlertLevel
from wandb.data_types import Table
from wandb.sdk.wandb_run import Run

from textbox.config.configurator import Config

train_step = 'train/step'
train_epoch = 'train/epoch'
valid_step = 'valid/step'
valid_epoch = 'valid/epoch'
metrics_labels = (train_step, train_epoch, valid_step, valid_epoch)

logger = getLogger(__name__)


class Timestamp:

    """A timestamp class including train step, train epoch, validation step and validation epoch."""

    def __init__(self):
        self.train_step = 0
        self.train_epoch = 0
        self.valid_step = 0
        self.valid_epoch = 0

    def update_axe(self, *name: str):
        """Update one of the timestamp."""
        axe = '_'.join(name)
        value = getattr(self, axe)
        if isinstance(value, int):
            setattr(self, axe, value+1)
        else:
            getLogger(__name__).warning(f'Failed when updating axe {axe}')

    def as_dict(self) -> dict:
        """Get the timestamp as a dictionary. The entries are also metrics labels shown in W&B."""
        return {
            train_step: self.train_step, train_epoch: self.train_epoch,
            valid_step: self.valid_step, valid_epoch: self.train_epoch,
        }


class EpochTracker:
    r"""Track the result of an epoch.

    Args:
        metrics_for_best_model: A set of entry to calculate validation score.
        mode: (default = None) "train" or "valid" mode.
        axes: (default = None) The timestamp for the moment.
    """

    def __init__(
            self,
            metrics_for_best_model: Optional[Set[str]] = None,
            mode: Optional[str] = None,
            axes: Optional[Timestamp] = None,
            metrics_results: Optional[dict] = None,
    ):

        # loss
        self._avg_loss: float = 0.
        self._accumulate_step: int = 0

        # metrics
        self._metrics_results: Dict[str, float] = dict()
        self.is_best_valid = False

        # result: loss & metrics
        self._score: Optional[float] = None

        self._start_time: float = -math.inf
        self._end_time: float = math.inf
        self.mode = mode or 'Epoch'
        self.axes = axes if axes is not None else Timestamp()
        self.metrics_for_best_model = metrics_for_best_model or set()
        if self.mode == 'train':
            self.desc = 'Train epoch '
            self.serial = self.axes.train_epoch
        elif self.mode == 'valid':
            self.desc = ' Validation '
            self.serial = self.axes.valid_epoch
        else:
            self.desc = 'Epoch '
            self.serial = ''
        self._has_score = False
        self._has_loss = False

        if metrics_results is not None:
            self._update_metrics(metrics_results)

    def _on_epoch_start(self):
        """Call at epoch start."""
        self._start_time = time()

    def _on_epoch_end(self, current_best: Optional[bool] = None):
        """Call at epoch end. This function will output information of this epoch."""
        self._end_time = time()
        _current_best = current_best if current_best is not None else self.is_best_valid
        self.epoch_info(self._end_time - self._start_time, _current_best)

    def _append_loss(self, loss: float):
        """Append the loss of one step."""
        self._avg_loss *= self._accumulate_step / (self._accumulate_step + 1)
        self._avg_loss += loss / (self._accumulate_step + 1)
        self._accumulate_step += 1
        self._has_loss = True

    @property
    def avg_loss(self) -> float:
        """Get the average of losses in each step."""
        if self._accumulate_step == 0:
            logger.warning("Trying to access epoch average loss before append any.")
            return math.inf
        return self._avg_loss

    def _update_metrics(self, results: Optional[dict] = None, **kwargs):
        """Update metrics result of this epoch."""
        for results_dict in (results, kwargs):
            if results_dict is not None:
                self._metrics_results.update(results_dict)
                self._has_score = True

    def as_dict(self) -> dict:
        """Return the epoch result as a dict"""
        results = {}
        if self._has_loss:
            results.update(loss=self.avg_loss)
        if self._has_score:
            results.update(score=self.calc_score())
            results.update(self._metrics_results)
        return results

    def calc_score(self) -> float:
        """calculate the total score of valid metrics for early stopping.

        If `loss` is in `keys`, the negative of average loss will be returned.
        Else, the sum of metrics results indexed by keys will be returned.
        """

        if 'loss' in self.metrics_for_best_model:
            return -self.avg_loss
        if 'score' in self._metrics_results:
            return self._metrics_results['score']

        score = 0.
        for metric, result in self._metrics_results.items():
            if metric in self.metrics_for_best_model and isinstance(result, float):
                score += result

        return score

    def _add_metric(self, _k: str, _v: Union[str, float], indent, sep) -> str:
        if isinstance(_v, str):
            return ''
        _o = indent
        if _k.lower() in self.metrics_for_best_model:
            _o += '<'
        if isinstance(_v, float):
            _o += f'{_k}: {_v:.2f}'
        if _k.lower() in self.metrics_for_best_model:
            _o += '>'
        return _o + sep

    def as_str(self, sep=', ', indent='') -> str:
        output = ''
        for metric, result in self.as_dict().items():
            output += self._add_metric(metric, result, indent, sep)
        return output[:-len(sep)]

    def epoch_info(
            self,
            time_duration: float,
            current_best: bool = False,
            desc: Optional[str] = None,
            serial: Union[int, str, None] = None,
            source: Optional[Callable] = None
    ):
        r"""Output loss with epoch and time information."""

        if serial is None:
            serial = self.serial
        output = "{} {} ".format(desc or self.desc, serial)
        if current_best:
            output += '(best) '
        output += f"[time: {time_duration:.2f}s, {self.as_str()}]"

        if source is None:
            source = logger.info
        source(output)

    def __repr__(self):
        return self.as_str()


class SummaryTracker:
    """Track the result of an experiment, including uploading result to W&B, calculating validation score
    and maintaining best result.

    Args:
        kwargs: The arguments of `wandb.init()`
        is_local_main_process: Decide whether to upload result
        metrics_for_best_model: A set of entry to calculate validation score.
        email: Whether to send an email through W&B.

    Examples:
        >>> summary_tracker = SummaryTracker.basicConfig(config)
        >>> with summary_tracker.new_experiment():
        >>>     # at the beginning of a validation epoch
        >>>     with summary_tracker.new_epoch('valid'):
        >>>         for step in valid_data:
        >>>             summary_tracker.new_step()
        >>>             ...
        >>>             summary_tracker.append_loss(1.0)
        >>>         summary_tracker.set_metrics_results({'metric-1': 1.0, 'metric-2': 2.0})
        >>>         best_valid_timestamp, current_best = summary_tracker.get_best_valid()
    """

    # wandb configurations
    _email = False
    _is_local_main_process = False
    tracker_finished = False
    _tables: Dict[str, Table] = dict()
    kwargs = dict()

    # experiment information
    axes = Timestamp()
    best_valid_score = -math.inf
    best_valid_timestamp = None
    metrics_for_best_model = set()

    # stack of current epochs
    _current_epoch: List[EpochTracker] = []
    _current_mode: List[str] = []
    _fallback: Optional[Tuple[EpochTracker, str]] = None
    is_best_valid: Optional[bool] = None

    def __init__(
            self,
            kwargs: dict,
            is_local_main_process: bool,
            metrics_for_best_model: Set[str],
            email: bool = True,
    ):
        self.__init_config(kwargs, is_local_main_process, metrics_for_best_model, email)

    @classmethod
    def __init_config(
        cls,
        kwargs: dict,
        is_local_main_process: bool,
        metrics_for_best_model: Set[str],
        email: bool = True,
    ):
        # wandb configurations
        cls._email = email
        cls._is_local_main_process = is_local_main_process
        cls.tracker_finished = False
        cls.kwargs = kwargs

        cls.metrics_for_best_model: Set[str] = metrics_for_best_model

    @classmethod
    def basicConfig(cls, config: Config):
        r"""Initialize dashboard configuration.

        Args:
            config: Configuration.

        Examples:
            >>> summary_tracker = SummaryTracker.basicConfig(config)
            >>> with summary_tracker.new_experiment():
            >>>     # at the beginning of a validation epoch
            >>>     with summary_tracker.new_epoch('valid'):
            >>>         for step in valid_data:
            >>>             summary_tracker.new_step()
            >>>             ...
            >>>             summary_tracker.append_loss(1.0)
            >>>         summary_tracker.set_metrics_results({'metric-1': 1.0, 'metric-2': 2.0})
            >>>         best_valid_timestamp, current_best = summary_tracker.get_best_valid()
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        global root
        if root is not None:
            return root

        project = f"{config['model']}-{config['dataset']}"
        name = config['filename'][len(project) + 1:]

        root = SummaryTracker(
            email=config['email'],
            is_local_main_process=config['_is_local_main_process'],
            metrics_for_best_model=config['metrics_for_best_model'],
            kwargs=dict(
                dir=config['logdir'],
                project=project,
                name=name,
                config=config.final_config_dict,
                # mode='disabled' if config['quick_test'] else 'online'
            )
        )
        return root

    @classmethod
    def current_epoch(cls, fallback: bool = False) -> EpochTracker:
        if len(cls._current_epoch) <= 0:
            if not fallback or cls._fallback is None:
                raise RuntimeError('No avaliable epoch information found. '
                                   'Move your code into `with new_epoch()` context to fix it.')
            else:
                return cls._fallback[0]
        return cls._current_epoch[-1]

    @classmethod
    def current_mode(cls, fallback: bool = False) -> str:
        if len(cls._current_epoch) <= 0:
            if not fallback or cls._fallback is None:
                raise RuntimeError('No avaliable epoch information found. '
                                   'Move your code into `with new_epoch()` context to fix it.')
            else:
                return cls._fallback[1]
        return cls._current_mode[-1]

    @classmethod
    @contextmanager
    def new_experiment(cls, reinit=False):
        _run = None
        if cls._is_local_main_process:
            _run = wandb.init(reinit=reinit, **cls.kwargs)
            wandb.define_metric("loss/train")
            wandb.define_metric("loss/valid")
            wandb.define_metric("metrics/*")

        cls.axes = Timestamp()

        cls.best_valid_score = -math.inf
        cls.best_valid_timestamp = Timestamp()
        cls.is_best_valid = None

        try:
            yield True

        except Exception:
            if cls._email:
                config = cls.kwargs['config']
                wandb.alert(title=f"Error {config['model']}-{config['dataset']}", 
                            text=f"{config['cmd']}\n\n{traceback.format_exc()}",
                            level=AlertLevel.ERROR)
            logger.error(traceback.format_exc())
            
        else:
            if cls._is_local_main_process and _run is not None:
                if cls._email:
                    try:
                        test_result = cls.current_epoch(fallback=True).as_str()
                    except RuntimeError:
                        test_result = 'None'
                    wandb.alert(title=f"Finished {cls.kwargs['project']}",
                                text=f"{cls.kwargs['config']['cmd']}\n\n{test_result}")
                cls.flush_text()
                _run.finish()
            cls.tracker_finished = True

        return root

    @classmethod
    @contextmanager
    def new_epoch(cls, mode: str):
        """Decorate the epoch function

        Args:
            mode: the mode of current epoch (train or valid)
        """

        if cls.axes is None:
            raise RuntimeError('You should decorate the function of experiment with new_experiment!')

        cls._current_mode.append(mode)
        if mode == 'train' or mode == 'valid':
            cls.axes.update_axe(mode, 'epoch')
        cls._current_epoch.append(EpochTracker(cls.metrics_for_best_model, mode=mode, axes=cls.axes))
        cls._current_epoch[-1]._on_epoch_start()

        yield True

        cls._current_epoch[-1]._on_epoch_end()
        cls._fallback = (cls._current_epoch[-1], cls._current_mode[-1])
        del cls._current_mode[-1]
        del cls._current_epoch[-1]


    @classmethod
    def new_step(cls, fallback: bool = False):
        r"""Call at the beginning of one step."""
        cls.axes.update_axe(cls.current_mode(fallback=fallback), 'step')

    @classmethod
    def append_loss(cls, loss: Union[float, torch.Tensor], fallback: bool = False):
        r"""Append loss of current step to tracker and update current step.

        Notes:
            `loss` should be a float! (like `loss.item()`)
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if math.isnan(loss):
            raise ValueError('Value is nan.')
        cls.add_scalar("loss/" + cls.current_mode(fallback=fallback), loss)
        cls.current_epoch(fallback=fallback)._append_loss(loss)

    @classmethod
    def epoch_loss(cls, fallback: bool = False) -> float:
        r"""Loss of this epoch. Average loss will be calculated and returned.
        """
        return cls.current_epoch(fallback=fallback).avg_loss

    @classmethod
    def set_metrics_results(cls, results: Optional[dict], fallback: bool = False):
        r"""Record the metrics results."""
        if results is None:
            return
        tag = 'metrics/' if cls.current_mode(fallback=fallback) != 'eval' else 'test/'
        for metric, result in results.items():
            cls.add_any(tag + metric, result)
            if not isinstance(result, str):
                cls.current_epoch(fallback=fallback)._update_metrics({metric: result})

        cls.is_best_valid = False
        if cls.epoch_score() > cls.best_valid_score:
            cls.best_valid_score = cls.epoch_score()
            cls.best_valid_timestamp = copy(cls.axes)
            cls.is_best_valid = True
            cls.current_epoch(fallback=fallback).is_best_valid = True

    @classmethod
    def epoch_score(cls, fallback: bool = False) -> float:
        r"""Get the score of current epoch calculated by `metrics_for_best_model`.

        Notes:
            If `loss` metric is in `metrics_for_best_model`, the negative of `loss` will be returned.
        """
        return cls.current_epoch(fallback=fallback).calc_score()

    @classmethod
    def epoch_dict(cls, fallback: bool = False) -> dict:
        r"""Get result of current epoch."""
        return cls.current_epoch(fallback=fallback).as_dict()

    @classmethod
    def add_text(cls, tag: str, text_string: str):
        r"""Add text to summary. The text will automatically upload to W&B at the end of experiment
        with `on_experiment_end()`. It may also be manually upload with `flush_text()`"""
        if cls._is_local_main_process and cls.axes is not None:
            if tag not in cls._tables:
                cls._tables[tag] = wandb.Table(columns=[train_step, tag])
            cls._tables[tag].add_data(cls.axes.train_step, text_string)

    @classmethod
    def flush_text(cls):
        r"""Manually flush temporary text added."""
        wandb.log(cls._tables)
        cls._tables = dict()

    @classmethod
    def add_scalar(cls, tag: str, scalar_value: Union[float, int]):
        r"""Add a scalar (`float` or `int`) to summary."""
        info = {tag: scalar_value}
        # info.update(self.axes.as_dict())
        if cls._is_local_main_process and not cls.tracker_finished and cls.axes is not None:
            wandb.log(info, step=cls.axes.train_step, commit=False)

    @classmethod
    def add_any(cls, tag: str, any_value: Union[str, float, int]):
        r"""Add a metric of any type (`float`, `int` and `str` supported) to summary."""
        if isinstance(any_value, str):
            cls.add_text(tag, any_value)
        elif isinstance(any_value, (float, int)):
            cls.add_scalar(tag, any_value)

    @classmethod
    def add_corpus(cls, tag: str, corpus: Iterable[str], fallback: bool = False):
        r"""Add a corpus to summary."""
        if tag.startswith('valid'):
            cls.current_epoch(fallback=fallback)._update_metrics({'generated_corpus': '\n'.join(corpus)})
        if cls._is_local_main_process and not cls.tracker_finished:
            _corpus = wandb.Table(columns=[tag], data=pd.DataFrame(corpus))
            wandb.log({tag: _corpus}, step=cls.axes.train_step)


root = None

def get_dashboard():
    """Return the root dashboard."""
    return root

