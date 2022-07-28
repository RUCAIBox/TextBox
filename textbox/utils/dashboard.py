"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`
"""
import math
import os
from copy import copy
from time import time

import pandas as pd
import torch
from logging import getLogger, Logger
import wandb

from typing import Optional, Union, Iterable, Dict, Collection, Set, Tuple
from textbox.config.configurator import Config

train_step = 'train/step'
train_epoch = 'train/epoch'
valid_step = 'valid/step'
valid_epoch = 'valid/epoch'
metrics_labels = (train_step, train_epoch, valid_step, valid_epoch)

MetricsDict = Dict[str, Union[float, Dict[str, float], Collection[float]]]


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


class SummaryTracker:
    """Track the result of an experiment, including uploading result to W&B, calculating validation score
    and maintaining best result.

    Args:
        kwargs: The arguments of `wandb.init()`
        is_local_main_process: Decide whether to upload result
        metrics_for_best_model: A set of entry to calculate validation score.
        email: Whether to send an email through W&B.

    Examples:
        >>> init_dashboard(config)
        >>> ...
        >>> start_dashboard()
        >>> summary_tracker = get_dashboard()
        >>> ...
        >>> # at the beginning of a validation epoch
        >>> summary_tracker.new_epoch('valid')
        >>> for step in valid_data:
        >>>     summary_tracker.new_step()
        >>>     ...
        >>>     summary_tracker.append_loss(1.0)
        >>> summary_tracker.set_metrics_results({'metric-1': 1.0, 'metric-2': 2.0})
        >>> best_valid_timestamp, current_best = summary_tracker.get_best_valid()
        >>> summary_tracker.on_epoch_end()
        >>> ...
        >>> finish_dashboard()
    """

    def __init__(
            self,
            kwargs: dict,
            is_local_main_process: bool,
            metrics_for_best_model: Set[str],
            email: bool = True,
    ):
        self._email = email
        self._is_local_main_process = is_local_main_process
        self.tracker_finished = False
        self.metrics_for_best_model: Set[str] = metrics_for_best_model
        self._tables: Dict[str, wandb.data_types.Table] = dict()
        self.kwargs = kwargs

        self._run = None
        self.axes = None

        self.current_epoch: Optional[EpochTracker] = None
        self.current_mode: Optional[str] = None

        self.best_valid_score = None
        self.best_valid_timestamp = None
        self.current_best = None

    def on_experiment_start(self):
        r"""Call at the beginning of experiment."""
        if self._is_local_main_process:
            self._run = wandb.init(reinit=True, **self.kwargs)
            for axe in metrics_labels:
                wandb.define_metric(axe)
            wandb.define_metric("loss/train", step_metric=train_step)
            wandb.define_metric("loss/valid", step_metric=train_step)
            wandb.define_metric("metrics/*", step_metric=train_step)

        self.axes = Timestamp()

        self.best_valid_score = -math.inf
        self.best_valid_timestamp = Timestamp()
        self.current_best = False

    def new_epoch(self, mode: str):
        r""" Call at the beginning of one epoch.
        Args:
            mode: Literal of "train" or "valid"
        """
        self.current_mode = mode
        self.axes.update_axe(mode, 'epoch')
        self.current_epoch = EpochTracker(self.metrics_for_best_model, mode=mode, axes=self.axes)
        self.current_epoch.on_epoch_start()

    def new_step(self):
        r"""Call at the beginning of one step."""
        if self.current_mode is None:
            raise RuntimeError('`new_epoch()` should be called before `new_step()`')
        self.axes.update_axe(self.current_mode, 'step')

    def append_loss(self, loss: Union[float, torch.Tensor]):
        r"""Append loss of current step to tracker and update current step."""
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if math.isnan(loss):
            raise ValueError('Value is nan.')
        self.add_scalar("loss/" + self.current_mode, loss)

        self.current_epoch.append_loss(loss)

    @property
    def epoch_loss(self) -> float:
        r"""Loss of this epoch. Average loss will be calculated and returned.
        """
        return self.current_epoch.avg_loss

    def set_metrics_results(self, results: dict):
        r"""Record the metrics results."""
        for metric, result in results.items():
            self.add_any('metrics/' + metric, result)
            if not isinstance(result, str):
                self.current_epoch.update_metrics({metric: result})

    @property
    def epoch_score(self) -> float:
        r"""Get the score of current epoch calculated by `metrics_for_best_model`.

        Notes:
            If `loss` metric is in `metrics_for_best_model`, the negative of `loss` will be returned.
        """
        return self.current_epoch.calc_score()

    def get_best_valid(self) -> Tuple[Timestamp, bool]:
        r"""Get the epoch index of the highest score.

        Returns:
            Tuple[Timestamp, bool]: A tuple of the best epoch timestamp and a boolean indicating
                whether the current epoch is the best
        """
        self.current_best = False
        if self.epoch_score > self.best_valid_score:
            self.best_valid_score = self.epoch_score
            self.best_valid_timestamp = copy(self.axes)
            self.current_best = True
        return self.best_valid_timestamp, self.current_best

    def epoch_dict(self) -> dict:
        r"""Get result of current epoch."""
        return self.current_epoch.as_dict()

    def on_epoch_end(self):
        r"""Call at the end of one epoch."""
        self.current_epoch.on_epoch_end(self.current_best if self.current_mode == 'valid' else False)
        self.current_mode = None

    def add_text(self, tag: str, text_string: str):
        r"""Add text to summary. The text will automatically upload to W&B at the end of experiment
        with `on_experiment_end()`. It may also be manually upload with `flush_text()`"""
        if self._is_local_main_process:
            if tag not in self._tables:
                self._tables[tag] = wandb.Table(columns=[train_step, tag])
            self._tables[tag].add_data(self.axes.train_step, text_string)

    def flush_text(self):
        r"""Manually flush temporary text added."""
        wandb.log(self._tables)
        self._tables = dict()

    def add_scalar(self, tag: str, scalar_value: Union[float, int]):
        r"""Add a scalar (`float` or `int`) to summary."""
        info = {tag: scalar_value}
        info.update(self.axes.as_dict())
        if self._is_local_main_process and not self.tracker_finished:
            wandb.log(info, step=self.axes.train_step)

    def add_any(self, tag: str, any_value: Union[str, float, int]):
        r"""Add a metric of any type (`float`, `int` and `str` supported) to summary."""
        if isinstance(any_value, str):
            self.add_text(tag, any_value)
        elif isinstance(any_value, (float, int)):
            self.add_scalar(tag, any_value)

    def add_corpus(self, tag: str, corpus: Iterable[str]):
        r"""Add a corpus to summary."""
        if tag.startswith('valid'):
            self.current_epoch.update_metrics({'generated_corpus': '\n'.join(corpus)})
        if self._is_local_main_process and not self.tracker_finished:
            corpus = wandb.Table(columns=[tag], data=pd.DataFrame(corpus))
            wandb.log({tag: corpus}, step=self.axes.train_step)

    def on_experiment_end(self):
        """Call at the end of experiment. `finish_dashboard()` will automatically call this function."""
        if self._is_local_main_process:
            if self._email:
                wandb.alert(title="Training Finished", text="The training is finished.")
            self.flush_text()
            self._run.finish()
        self.tracker_finished = True


class EpochTracker:
    r"""Track the result of an epoch.

    Args:
        metrics_for_best_model: A set of entry to calculate validation score.
        mode: (default = None) "train" or "valid" mode.
        axes: (default = None) The timestamp for the moment.
    """

    def __init__(
            self,
            metrics_for_best_model: Set[str],
            mode: Optional[str] = None,
            axes: Optional[Timestamp] = None,
    ):

        # loss
        self._avg_loss: float = 0.
        self._accumulate_step: int = 0

        # metrics
        self._valid_metrics_results: MetricsDict = dict()

        # result: loss & metrics
        self._score: Optional[float] = None

        self._logger: Logger = getLogger(__name__)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self.mode = mode or 'Epoch'
        self.axes = axes
        self.metrics_for_best_model = metrics_for_best_model
        if self.mode == 'train':
            self.desc = 'Train epoch '
            self.serial = self.axes.train_epoch
        elif self.mode == 'valid':
            self.desc = ' Validation '
            self.serial = self.axes.valid_epoch
        else:
            self.desc = 'Epoch '

    def on_epoch_start(self):
        """Call at epoch start."""
        self._start_time = time()

    def on_epoch_end(self, current_best: bool):
        """Call at epoch end. This function will output information of this epoch."""
        self._end_time = time()
        self._epoch_info(self._end_time - self._start_time, current_best)

    def append_loss(self, loss: float):
        """Append the loss of one step."""
        self._avg_loss *= self._accumulate_step / (self._accumulate_step + 1)
        self._avg_loss += loss / (self._accumulate_step + 1)
        self._accumulate_step += 1

    @property
    def avg_loss(self) -> float:
        """Get the average of losses in each step."""
        if self._accumulate_step == 0:
            self._logger.warning("Trying to access epoch average loss before append any.")
            return math.inf
        return self._avg_loss

    def update_metrics(self, results: Optional[dict] = None, **kwargs):
        """Update metrics result of this epoch."""
        if results is not None:
            self._valid_metrics_results.update(results)
        self._valid_metrics_results.update(kwargs)

    def as_dict(self) -> dict:
        """Return the epoch result as a dict"""
        if self._valid_metrics_results:
            results = self._valid_metrics_results
        else:
            results = {}
        if self._accumulate_step != 0:
            results.update(loss=self.avg_loss)
        if self.mode == 'valid':
            results.update(score=self.calc_score())
        return results

    def calc_score(self) -> float:
        """calculate the total score of valid metrics for early stopping.

        If `loss` is in `keys`, the negative of average loss will be returned.
        Else, the sum of metrics results indexed by keys will be returned.
        """

        if 'loss' in self.metrics_for_best_model:
            return -self.avg_loss

        score = 0.
        for metric, result in self._valid_metrics_results.items():
            if metric in self.metrics_for_best_model and isinstance(result, float):
                score += result

        return score

    def _epoch_info(self, time_duration: float, current_best: bool):
        r"""Output loss with epoch and time information."""
        def add_metric(_k: str, _v: Union[str, float]) -> str:
            if isinstance(_v, str):
                return ''
            _o = ', '
            if _k.lower() in self.metrics_for_best_model:
                _o += '<'
            if isinstance(_v, float):
                _o += f'{_k}: {_v:.4f}'
            if _k.lower() in self.metrics_for_best_model:
                _o += '>'
            return _o

        output = "{} {} ".format(self.desc, self.serial)
        if current_best:
            output += '(best) '
        output += "[time: {:.2f}s".format(time_duration)
        if self.mode == 'valid':
            output += add_metric('score', self.calc_score())

        for metric, result in self.as_dict().items():
            if isinstance(result, dict):
                for key, value in result.items():
                    output += add_metric(key, value)
            else:
                output += add_metric(metric, result)

        output += "]"

        self._logger.info(output)


root: Optional[SummaryTracker] = None


def init_dashboard(
        config: Config,
) -> SummaryTracker:
    r"""Initialize dashboard configuration.

    Args:
        config: Configuration.

    Notes:
        After initialize configuration, call `start_dashboard()` and `finish_dashboard()`
        to start and finish one run.

    Examples:
        >>> init_dashboard(config)
        >>> ...
        >>> start_dashboard()
        >>> summary_tracker = get_dashboard()
        >>> ...
        >>> # at the beginning of a validation epoch
        >>> summary_tracker.new_epoch('valid')
        >>> for step in valid_data:
        >>>     summary_tracker.new_step()
        >>>     ...
        >>>     summary_tracker.append_loss(1.0)
        >>> summary_tracker.set_metrics_results({'metric-1': 1.0, 'metric-2': 2.0})
        >>> best_valid_timestamp, current_best = summary_tracker.get_best_valid()
        >>> summary_tracker.on_epoch_end()
        >>> ...
        >>> finish_dashboard()
    """
    os.environ["WANDB_SILENT"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    global root
    if root is not None:
        return root

    project = f"{config['model']}-{config['dataset']}"
    name = config['filename'][len(project) + 1:]

    root = SummaryTracker(
        email=config['email'],
        is_local_main_process=config['is_local_main_process'],
        metrics_for_best_model=config['metrics_for_best_model'],
        kwargs=dict(
            dir=config['logdir'],
            project=project,
            name=name,
            config=config.final_config_dict,
            mode='disabled' if config['quick_test'] else 'online'
        )
    )

    return root


def start_dashboard():
    root.on_experiment_start()


def finish_dashboard():
    """Close dashboard tracking."""
    root.on_experiment_end()


def get_dashboard() -> SummaryTracker:
    """Get summary tracker. `init_dashboard()` must be called before `get_dashboard()`"""
    assert root is not None, "Please initialize dashboard first."
    return root
