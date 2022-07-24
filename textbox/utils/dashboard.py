"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`

Todo:
    * WandBWriter with distributed training
    * wandb: resume?
"""
import math
import os
from time import time

import pandas as pd
import torch
from logging import getLogger, Logger
import wandb

from typing import Optional, Union, Iterable, Dict, Collection, Set
from textbox.config.configurator import Config

train_step = 'train/step'
train_epoch = 'train/epoch'
valid_step = 'valid/step'
valid_epoch = 'valid/epoch'
axes_label = (train_step, train_epoch, valid_step, valid_epoch)

MetricsDict = Dict[str, Union[float, Dict[str, float], Collection[float]]]


class SummaryTracker:

    def __init__(
            self,
            kwargs: dict,
            is_local_main_process: bool,
            metrics_for_best_model: Set[str],
            email: bool = True,
    ):
        self._axes = dict.fromkeys(axes_label, 0)
        self._email = email
        self._is_local_main_process = is_local_main_process
        self.tracker_finished = False
        self.metrics_for_best_model: Set[str] = metrics_for_best_model
        self._run = None
        self._tables: Dict[str, wandb.data_types.Table] = dict()
        self.kwargs = kwargs

        self.current_epoch: Optional[EpochTracker] = None
        self.current_mode: Optional[str] = None

    def on_experiment_start(self):
        if self._is_local_main_process:
            self._run = wandb.init(reinit=True, **self.kwargs)
            for axe in axes_label:
                wandb.define_metric(axe)
            wandb.define_metric("loss/train", step_metric=train_step)
            wandb.define_metric("loss/valid", step_metric=train_step)
            wandb.define_metric("metrics/*", step_metric=train_step)

    def new_epoch(self, mode: str):
        self.current_mode = mode
        axe = mode + '/epoch'
        self.update_axe(axe)
        self.current_epoch = EpochTracker(mode, self._axes[axe], self.metrics_for_best_model)
        self.current_epoch.on_epoch_start()

    def append_loss(self, loss: Union[float, torch.Tensor]):
        r"""Append loss of current step to tracker and update current step."""
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if math.isnan(loss):
            raise ValueError('Value is nan.')
        self.add_scalar("loss/" + self.current_mode, loss)
        self.update_axe(self.current_mode + "/step")

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
        return self.current_epoch.calc_score()

    def epoch_dict(self) -> dict:
        return self.current_epoch.as_dict()

    def update_axe(self, axe: str):
        if axe not in self._axes:
            getLogger().warning(f'Failed when updating axe {axe}.')
        else:
            self._axes[axe] += 1

    def on_epoch_end(self):
        self.current_epoch.on_epoch_end()

    def add_text(self, tag: str, text_string: str):
        if self._is_local_main_process:
            if tag not in self._tables:
                self._tables[tag] = wandb.Table(columns=[train_step, tag])
            self._tables[tag].add_data(self._axes[train_step], text_string)

    def flush_text(self):
        wandb.log(self._tables)
        self._tables = dict()

    def add_scalar(self, tag: str, scalar_value: Union[float, int]):
        info = {tag: scalar_value}
        info.update(self._axes)
        if self._is_local_main_process and not self.tracker_finished:
            wandb.log(info, step=self._axes['train/step'])

    def add_any(self, tag: str, any_value: Union[str, float, int]):
        if isinstance(any_value, str):
            self.add_text(tag, any_value)
        elif isinstance(any_value, (float, int)):
            self.add_scalar(tag, any_value)

    def add_corpus(self, tag: str, corpus: Iterable[str]):
        if self._is_local_main_process and not self.tracker_finished:
            corpus = wandb.Table(columns=[tag], data=pd.DataFrame(corpus))
            wandb.log({tag: corpus}, step=self._axes[train_step])

    def on_experiment_end(self):
        if self._is_local_main_process:
            if self._email:
                wandb.alert(title="Training Finished", text="The training is finished.")
            self.flush_text()
            self._run.finish()
        self.tracker_finished = True


class EpochTracker:
    r"""
    Track and visualize validating metrics results and training / validating loss.
    Dashboard now supports :class:`textbox.utils.TensorboardWriter`.

    Args:
        epoch_idx: Current epoch index.
        mode: Optional (default = True)
            Train or eval mode.

    Example:
        >>> tbw = TensorboardWriter("log/dir", "filename")
        >>> for epoch in range(10):
        >>>     valid_tracker = EpochTracker(epoch, train=False, dashboard=tbw)
        >>>     for step in range(10):
        >>>         valid_tracker.append_loss(1.0)
        >>>     valid_tracker.set_result({"metric1": 1.0})
        >>>     valid_tracker.info(time_duration=10.0)
        Epoch 0,  validating [time: 10.00s, validating_loss: 1.0000]

    Todo:
        * Update docstring
    """

    def __init__(self, mode: str, epoch_idx: int, metrics_for_best_model: Set[str]):

        # loss
        self._avg_loss: float = 0.
        self._accumulate_step: int = 0
        self._reduced_loss: Optional[float] = None

        # metrics
        self._valid_metrics_results: MetricsDict = dict()

        # result: loss & metrics
        self._score: Optional[float] = None

        self._logger: Logger = getLogger()
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self.mode_tag = mode
        self.epoch_idx = epoch_idx
        self.metrics_for_best_model = metrics_for_best_model

    def on_epoch_start(self):
        self._start_time = time()

    def on_epoch_end(self):
        self._end_time = time()
        if self._accumulate_step != 0:
            self._loss_all_reduce()
        self._epoch_info(self._end_time - self._start_time)

    def append_loss(self, loss: float):
        self._avg_loss *= self._accumulate_step / (self._accumulate_step + 1)
        self._avg_loss += loss / (self._accumulate_step + 1)
        self._accumulate_step += 1

    @property
    def avg_loss(self) -> float:
        if self._accumulate_step == 0:
            self._logger.warning("Trying to access epoch average loss before append any.")
            return math.inf
        if self._reduced_loss is not None:
            return self._reduced_loss
        else:
            return self._avg_loss

    def update_metrics(self, results: Optional[dict] = None, **kwargs):
        if results is not None:
            self._valid_metrics_results.update(results)
        self._valid_metrics_results.update(kwargs)

    def as_dict(self) -> dict:
        if self._valid_metrics_results:
            results = self._valid_metrics_results
        else:
            results = {}
        if self._accumulate_step != 0:
            results.update(loss=self.avg_loss)
        if self.mode_tag == 'valid':
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
        float_list = []
        for metric, result in self._valid_metrics_results.items():
            if isinstance(result, dict):
                float_list = [v for k, v in result.items() if k in self.metrics_for_best_model and isinstance(v, float)]
            elif metric in self.metrics_for_best_model:
                if isinstance(result, Collection):
                    float_list = list(filter(lambda x: isinstance(x, float), result))
                elif isinstance(result, float):
                    float_list = [result]

            if len(float_list) != 0:
                score += sum(float_list) / len(float_list)
                float_list = []

        return score

    def _loss_all_reduce(self):
        pass

    def _epoch_info(self, time_duration):
        r"""Output loss with epoch and time information."""
        def add_metric(_k, _v):
            _o = ', '
            if _k.lower() in self.metrics_for_best_model:
                _o += '<'
            if isinstance(_v, str):
                _o += f'{_k}: "{_v[:15]}..."'
            elif isinstance(_v, float):
                _o += f'{_k}: {_v:.4f}'
            if _k.lower() in self.metrics_for_best_model:
                _o += '>'
            return _o

        output = "Epoch {}, {} [time: {:.2f}s".format(self.epoch_idx, self.mode_tag, time_duration)

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
    r"""Initialize configuration of W&B.

    Notes:
        After initialize configuration, call `start_dashboard()` and `finish_dashboard()`
        to start and finish one run.
    """
    os.environ["WANDB_SILENT"] = "true"

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
            config=config.final_config_dict
        )
    )

    return root


def start_dashboard():
    root.on_experiment_start()


def finish_dashboard():
    root.on_experiment_end()


def get_dashboard() -> SummaryTracker:
    assert root is not None, "Please initialize dashboard first."
    return root
