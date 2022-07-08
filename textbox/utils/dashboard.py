"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`

Todo:
    * WandBWriter with distributed training
    * wandb: resume?
"""

import os
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None
    getLogger().info('Failed when importing wandb module.')

from typing import Optional, Any, Type, Union, Iterable, Callable, Dict
from textbox.config.configurator import Config
ScalarType = (float, int)
TextType = str


class AbstractDashboard:

    def __init__(self):
        self._axes_label = ['train/step', 'train/epoch', 'valid/step', 'valid/epoch']
        self._axes = dict.fromkeys(self._axes_label, 0)

    def update_axes(self, *args: str):

        for axe in args:
            if axe not in self._axes:
                getLogger().warning(f'Failed when updating axe {axe}.')
            else:
                self._axes[axe] += 1

    def __enter__(self):

        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):

        raise NotImplementedError

    def add_scalar(self, tag: str, scalar_value: Union[ScalarType], **kwargs):
        r"""Add scalar data to summary.

        Args:
            tag: Data identifier
            scalar_value: Value to save
            **kwargs
        """
        raise NotImplementedError

    def add_text(self, tag: str, text_string: Union[TextType], **kwargs):
        r"""Add text data to summary.

        Args:
            tag: Data identifier
            text_string: Value to save
            **kwargs
        """
        raise NotImplementedError

    def add_dict(self, tag: str, dictionary: dict, **kwargs):
        if not tag.endswith("/"):
            tag += "/"
        for key, value in dictionary:
            if isinstance(value, ScalarType):
                self.add_scalar(tag + key, value, **kwargs)
            elif isinstance(value, TextType):
                self.add_text(tag + key, value, **kwargs)
            else:
                getLogger().info(f"Received not supported element `{key}: {value}` when adding dictionary {tag}.")

    def add_iterable(self, tag: str, iterable: Iterable, **kwargs):
        if not tag.endswith("/"):
            tag += "/"
        for idx, value in enumerate(iterable):
            if isinstance(value, ScalarType):
                self.add_scalar(tag + str(idx), value, **kwargs)
            elif isinstance(value, TextType):
                self.add_text(tag + str(idx), value, **kwargs)
            else:
                getLogger().info(f"Received not supported element `{value}` at index {idx} when adding iterable {tag}.")

    def add_any(self, tag: str, any_value: Any, **kwargs):
        if isinstance(any_value, TextType):
            self.add_text(tag, any_value, **kwargs)
        elif isinstance(any_value, ScalarType):
            self.add_scalar(tag, any_value, **kwargs)
        elif isinstance(any_value, dict):
            self.add_dict(tag, any_value, **kwargs)
        elif isinstance(any_value, Iterable):
            self.add_iterable(tag, any_value, **kwargs)


class NilWriter(AbstractDashboard):

    def __init__(self):
        super().__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        r"""Dummy method."""
        pass

    def add_scalar(self, tag: str, scalar_value: Union[ScalarType], **kwargs):
        r"""Dummy method."""
        pass

    def add_text(self, tag: str, text_string: Union[TextType], **kwargs):
        r"""Dummy method."""
        pass


class TensorboardWriter(AbstractDashboard):
    """
    Tensorboard :class:`SummaryWriter` adapter.

    Example:
        >>> with TensorboardWriter("log/dir") as tbw:
        >>>     for epoch in range(10):
        >>>         ...
        >>>         tbw.add_scalar("Tag", 1)
        >>>         tbw.add_scalar("Another/Tag", 2)
        >>>         tbw.update_axes("train/step")
        >>>         ...
    """

    def __init__(self, logdir: str, filename: str):
        logdir = os.path.join(logdir, "tensorboard")
        log_subdir = os.path.join(logdir, filename)

        super(TensorboardWriter, self).__init__()

        self.dashboard = SummaryWriter(log_dir=log_subdir)

        print(
            f'Open dashboard with following command:\n\n  '
            f'tensorboard --logdir={os.path.abspath(logdir)} --host=127.0.0.1\n'
        )

    def __enter__(self) -> "TensorboardWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dashboard.flush()
        self.dashboard.close()

    def add_scalar(self, tag, scalar_value, **kwargs):
        self.dashboard.add_scalar(tag, scalar_value, global_step=self._axes['train/step'], **kwargs)

    def add_text(self, tag, text_string, **kwargs):
        self.dashboard.add_text(tag, text_string, global_step=self._axes['train/step'], **kwargs)


class WandBWriter(AbstractDashboard):

    def __init__(self, email=True, **kwargs):
        super().__init__()
        self.email = email
        self.run = wandb.init(**kwargs)
        wandb.define_metric("train/epoch")
        wandb.define_metric("valid/epoch")
        wandb.define_metric('train/step')
        wandb.define_metric('valid/step')
        wandb.define_metric("loss/train", step_metric="train/step")
        wandb.define_metric("loss/valid", step_metric="train/step")
        wandb.define_metric("metrics/*", step_metric="train/step")
        self.tables: Dict[str, wandb.data_types.Table] = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.log(self.tables)
        if self.email:
            wandb.alert(title="Training Finished", text="The training is finished.")
        self.run.finish()

    def add_text(self, tag: str, text_string: Union[TextType], **kwargs):
        if tag not in self.tables:
            self.tables[tag] = wandb.Table(columns=[tag])
        self.tables[tag].add_data(text_string)

    def add_scalar(self, tag: str, scalar_value: Union[ScalarType], **kwargs):
        info = {tag: scalar_value}
        info.update(self._axes)
        wandb.log(info, step=self._axes['train/step'])


def get_dashboard(
        dashboard: Optional[str],
        logdir: str,
        config: Config,
) -> Union[Callable[[], WandBWriter], Callable[[], TensorboardWriter], Type[NilWriter]]:
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
    if dashboard == "wandb":
        if wandb is not None:

            project = f"{config['model']}-{config['dataset']}"
            name = config['filename'][len(project)+1:]

            def _get_wandb():
                return WandBWriter(
                    email=config['email'],
                    dir=logdir,
                    project=project,
                    name=name,
                    config=config.final_config_dict
                )

            return _get_wandb
        else:
            dashboard = "tensorboard"

    if dashboard is None or dashboard == "tensorboard":

        def _get_tensorboard():
            return TensorboardWriter(logdir=logdir, filename=config['filename'])

        return _get_tensorboard
    else:
        return NilWriter
