"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`
"""
from contextlib import contextmanager
from copy import copy
from accelerate.logging import get_logger
import math
import os
from time import time
from typing import Iterable, Optional, Set, Union, Callable
import traceback

import torch
import wandb
from wandb import AlertLevel
from textbox.config.configurator import Config

train_step = 'train/step'
train_epoch = 'train/epoch'
valid_step = 'valid/step'
valid_epoch = 'valid/epoch'
metrics_labels = (train_step, train_epoch, valid_step, valid_epoch)

logger = get_logger(__name__)


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
            setattr(self, axe, value + 1)
        else:
            get_logger(__name__).warning(f'Failed when updating axe {axe}')

    def as_dict(self) -> dict:
        """Get the timestamp as a dictionary. The entries are also metrics labels shown in W&B."""
        return {
            train_step: self.train_step,
            train_epoch: self.train_epoch,
            valid_step: self.valid_step,
            valid_epoch: self.train_epoch,
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
        self._avg_loss = 0.
        self._accumulate_step = 0

        # metrics
        self._metrics_results = dict()
        self.is_best_valid = False

        # result: loss & metrics
        self._score = None

        self._start_time = -math.inf
        self._end_time = math.inf
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
            results.update(loss=self._avg_loss)
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
            return -self._avg_loss
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
        source: Optional[Callable] = logger.info
    ):
        r"""Output loss with epoch and time information."""

        if serial is None:
            serial = self.serial
        output = "{} {} ".format(desc or self.desc, serial)
        if current_best:
            output += '(best) '
        output += f"[time: {time_duration:.2f}s, {self.as_str()}]"

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
    _tables = dict()
    kwargs = dict()

    # experiment information
    axes = Timestamp()
    best_valid_score = -math.inf
    best_valid_timestamp = None
    metrics_for_best_model = set()

    # stack of current epochs
    _current_epoch = None
    _current_mode = None
    is_best_valid = None

    def __init__(
        self,
        kwargs: dict,
        is_local_main_process: bool,
        metrics_for_best_model: Set[str],
        email: bool = True,
    ):
        # wandb configurations
        self._email = email
        self._is_local_main_process = is_local_main_process
        self.tracker_finished = False
        self.kwargs = kwargs

        self.metrics_for_best_model = metrics_for_best_model

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
        saved_dir = os.path.join(config['saved_dir'], config['filename'])

        root = SummaryTracker(
            email=config['email'],
            is_local_main_process=config['_is_local_main_process'],
            metrics_for_best_model=config['metrics_for_best_model'],
            kwargs=dict(
                dir=saved_dir,
                project=project,
                name=name,
                config=config.final_config_dict,
                # mode='disabled' if config['quick_test'] else 'online'
            )
        )
        return root

    @contextmanager
    def new_experiment(self, reinit=False):
        _run = None
        if self._is_local_main_process:
            _run = wandb.init(reinit=reinit, **self.kwargs)
            wandb.define_metric("loss/train")
            wandb.define_metric("loss/valid")
            wandb.define_metric("metrics/*")

        self.axes = Timestamp()

        self.best_valid_score = -math.inf
        self.best_valid_timestamp = Timestamp()
        self.is_best_valid = None

        try:
            yield True

        except Exception:
            if self._email:
                config = self.kwargs['config']
                wandb.alert(
                    title=f"Error {config['model']}-{config['dataset']}",
                    text=f"{config['cmd']}\n\n{traceback.format_exc()}",
                    level=AlertLevel.ERROR
                )
            logger.error(traceback.format_exc())

        else:
            if self._is_local_main_process and _run is not None:
                if self._email:
                    try:
                        test_result = self._last_epoch.as_str()
                    except RuntimeError:
                        test_result = 'None'
                    wandb.alert(
                        title=f"Finished {self.kwargs['project']}",
                        text=f"{self.kwargs['config']['cmd']}\n\n{test_result}"
                    )
                self.flush_text()
                _run.finish()
            self.tracker_finished = True

        return root

    @contextmanager
    def new_epoch(self, mode: str):
        """Decorate the epoch function

        Args:
            mode: the mode of current epoch (train or valid)
        """
        if self.axes is None:
            raise RuntimeError('You should decorate the function of experiment with new_experiment!')

        self._current_mode = mode
        if mode == 'train' or mode == 'valid':
            self.axes.update_axe(mode, 'epoch')
        self._current_epoch = EpochTracker(self.metrics_for_best_model, mode=mode, axes=self.axes)
        self._current_epoch._on_epoch_start()

        yield True

        self._current_epoch._on_epoch_end()
        self._last_epoch = self._current_epoch
        self._current_mode = None
        self._current_epoch = None

    def new_step(self):
        r"""Call at the beginning of one step."""
        self.axes.update_axe(self._current_mode, 'step')

    def append_loss(self, loss: Union[float, torch.Tensor]):
        r"""Append loss of current step to tracker and update current step.

        Notes:
            `loss` should be a float! (like `loss.item()`)
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if math.isnan(loss):
            raise ValueError('Value is nan.')
        self.add_scalar("loss/" + self._current_mode, loss)
        self._current_epoch._append_loss(loss)

    @property
    def epoch_loss(self) -> float:
        r"""Loss of this epoch. Average loss will be calculated and returned.
        """
        return self._current_epoch._avg_loss

    def set_metrics_results(self, results: Optional[dict]):
        r"""Record the metrics results."""
        if results is None:
            return
        tag = 'metrics/' if self._current_mode != 'eval' else 'test/'
        for metric, result in results.items():
            if isinstance(result, str):
                self.add_text(tag + metric, result)
            else:
                self.add_scalar(tag + metric, result)
                self._current_epoch._update_metrics({metric: result})

        self.is_best_valid = False
        if self.epoch_score() > self.best_valid_score:
            self.best_valid_score = self.epoch_score()
            self.best_valid_timestamp = copy(self.axes)
            self.is_best_valid = True
            self._current_epoch.is_best_valid = True

    def epoch_score(self) -> float:
        r"""Get the score of current epoch calculated by `metrics_for_best_model`.

        Notes:
            If `loss` metric is in `metrics_for_best_model`, the negative of `loss` will be returned.
        """
        return self._current_epoch.calc_score()

    def epoch_dict(self) -> dict:
        r"""Get result of current epoch."""
        return self._current_epoch.as_dict()

    def add_text(self, tag: str, text_string: str):
        r"""Add text to summary. The text will automatically upload to W&B at the end of experiment
        with `on_experiment_end()`. It may also be manually upload with `flush_text()`"""
        if self._is_local_main_process and self.axes is not None:
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
        # info.update(self.axes.as_dict())
        if self._is_local_main_process and not self.tracker_finished and self.axes is not None:
            wandb.log(info, step=self.axes.train_step, commit=False)


root = None


def get_dashboard():
    """Return the root dashboard."""
    return root
