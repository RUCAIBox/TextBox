"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`

Todo:
    * WandBWriter with distributed training
    * wandb: resume?
"""
import math
from time import time

import pandas as pd
import torch
from logging import getLogger, Logger
import wandb

from typing import Optional, List, Union, Iterable, Callable, Dict, Collection
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
            DDP: bool,
            kwargs: dict,
            is_local_main_process: bool,
            email: bool = True,
    ):
        self._axes = dict.fromkeys(axes_label, 0)
        self._email = email
        self._DDP = DDP
        self._is_local_main_process = is_local_main_process
        self.tracker_finished = False

        if (self._is_local_main_process):
            self._run = wandb.init(**kwargs)
            for axe in axes_label:
                wandb.define_metric(axe)
            wandb.define_metric("loss/train", step_metric=train_step)
            wandb.define_metric("loss/valid", step_metric=train_step)
            wandb.define_metric("metrics/*", step_metric=train_step)
            self._tables: Dict[str, wandb.data_types.Table] = dict()

        self.current_epoch: Optional[EpochTracker] = None
        self.current_mode: Optional[str] = None

    def new_epoch(self, mode: str):
        self.current_mode = mode
        axe = mode + '/epoch'
        self.update_axe(axe)
        self.current_epoch = EpochTracker(self._DDP, mode, self._axes[axe])
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

        Notes:
            If DDP is enabled and `loss_all_reduce()` has been called, average loss of all machines will be returned.
        """
        return self.current_epoch.avg_loss

    def set_metrics_results(self, results: dict):
        r"""Record the metrics results."""
        for metric, result in results.items():
            self.add_any('metrics/' + metric, result)
            if not isinstance(result, str):
                self.current_epoch.update_metrics(metric=result)

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
        if (self._is_local_main_process):
            if tag not in self._tables:
                self._tables[tag] = wandb.Table(columns=[train_step, tag])
            self._tables[tag].add_data(self._axes[train_step], text_string)

    def add_scalar(self, tag: str, scalar_value: Union[float, int]):
        info = {tag: scalar_value}
        info.update(self._axes)
        if (self._is_local_main_process):
            wandb.log(info, step=self._axes['train/step'])

    def add_any(self, tag: str, any_value: Union[str, float, int]):
        if isinstance(any_value, str):
            self.add_text(tag, any_value)
        elif isinstance(any_value, (float, int)):
            self.add_scalar(tag, any_value)

    def add_corpus(self, tag: str, corpus: Iterable[str]):
        if (self._is_local_main_process):
            corpus = wandb.Table(columns=[tag], data=pd.DataFrame(corpus))
            wandb.log({tag: corpus}, step=self._axes[train_step])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._email:
            wandb.alert(title="Training Finished", text="The training is finished.")
        if (self._is_local_main_process):
            wandb.log(self._tables)
            self._run.finish()
        self.tracker_finished = True


class EpochTracker:
    r"""
    Track and visualize validating metrics results and training / validating loss.
    Dashboard now supports :class:`textbox.utils.TensorboardWriter`.

    Args:
        epoch_idx: Current epoch index.
        DDP: Whether the Pytorch DDP is enabled.
        mode: Optional (default = True)
            Train or eval mode.

    Example:
        >>> tbw = TensorboardWriter("log/dir", "filename")
        >>> for epoch in range(10):
        >>>     valid_tracker = EpochTracker(epoch, DDP=False, train=False, dashboard=tbw)
        >>>     for step in range(10):
        >>>         valid_tracker.append_loss(1.0)
        >>>     valid_tracker.set_result({"metric1": 1.0})
        >>>     valid_tracker.info(time_duration=10.0)
        Epoch 0,  validating [time: 10.00s, validating_loss: 1.0000]
    """

    def __init__(self, DDP: bool, mode: str, epoch_idx: int):

        # loss
        self._avg_loss: float = 0.
        self._accumulate_step: int = 0
        self._reduced_loss: Optional[float] = None

        # metrics
        self._valid_metrics_results: MetricsDict = dict()

        # result: loss & metrics
        self._score: Optional[float] = None

        self._logger: Logger = getLogger()
        self._DDP: bool = DDP
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self.mode_tag = mode
        self.epoch_idx = epoch_idx

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
            return self._valid_metrics_results
        else:
            return {'loss': self.avg_loss}

    def calc_score(self, keys: Iterable[str] = []) -> float:

        # todo: how to calculate the score?

        # calculate the total score of valid metrics for early stopping.
        score: Optional[float] = None
        for metric, result in self._valid_metrics_results.items():
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
                if score is None:
                    score = 0.
                score += sum(float_list) / len(float_list)

        if score is None:
            score = -self.avg_loss  # If no valid metric is given, negative loss will be used in early stopping.

        return score

    def _loss_all_reduce(self):
        if self._DDP:
            _loss = torch.tensor(self.avg_loss, device="cuda")
            torch.distributed.all_reduce(_loss, op=torch.distributed.ReduceOp.SUM)
            _loss /= torch.distributed.get_world_size()
            self._reduced_loss = _loss.item()

    def _epoch_info(self, time_duration):
        r"""Output loss with epoch and time information."""
        output = "Epoch {}, {} [time: {:.2f}s, ".format(self.epoch_idx, self.mode_tag, time_duration)

        for metric, result in self.as_dict().items():
            if isinstance(result, str):
                output += f', {metric}: "{result[:15]}..."'
            elif isinstance(result, float):
                output += f', {metric}: {result:.4f}'

        output += "]"

        self._logger.info(output)


def get_dashboard(
        logdir: str,
        config: Config,
) -> Callable[[], SummaryTracker]:
    r"""Get the dashboard class.

    Args:
        dashboard: Name of dashboard (`tensorboard`, `wandb` or None).
        logdir: Directory of log files. Subdirectories of dashboards will be created.
        config: Configuration.

    Examples:
        >>> TB = get_dashboard("tensorboard", logdir="log/dir", config=config)
        >>> with TB() as tbw:
        >>>     for epoch in range(10):
        >>>         ...
        >>>         tbw.add_scalar("Tag", 1)
        >>>         tbw.add_scalar("Another/Tag", 2)
        >>>         tbw.update_axes("train/step")
        >>>         ...

    """
    project = f"{config['model']}-{config['dataset']}"
    name = config['filename'][len(project) + 1:]

    def _get_wandb():
        return SummaryTracker(
            DDP=config['DDP'],
            email=config['email'],
            is_local_main_process=config['is_local_main_process'],
            # is_local_main_process=True,
            kwargs=dict(
                dir=logdir,
                project=project,
                name=name,
                config=config.final_config_dict
            )
        )

    return _get_wandb
