"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`

Todo:
    * Implement WandBWriter
"""

import os
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger

import textbox

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None
    getLogger().info('Failed when importing wandb module.')

from typing import Optional, Any, Type, Union, Iterable, Callable, Mapping
from textbox.config.configurator import Config
ScalarType = (float, int)
TextType = str


class AbstractDashboard:

    def __init__(self):
        self.global_step: int = 0

    def step(self):
        self.global_step += 1

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

    def __init__(self, logdir: str = "", filename: str = ""):
        super().__init__(logdir, filename)

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
        >>>         tbw.step()
        >>>         ...
    """

    def __init__(self, logdir: str, filename: str, comment: str = ""):
        logdir = os.path.join(logdir, "tensorboard")
        log_subdir = os.path.join(logdir, filename)

        super(TensorboardWriter, self).__init__()

        self.dashboard = SummaryWriter(log_dir=log_subdir, comment=comment)

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
        self.dashboard.add_scalar(tag, scalar_value, global_step=self.global_step, **kwargs)

    def add_text(self, tag, text_string, **kwargs):
        self.dashboard.add_text(tag, text_string, global_step=self.global_step, **kwargs)


class WandBWriter(AbstractDashboard):

    def __init__(self, filename: str):
        super().__init__()
        print('Login to wandb with following command:\n\n  wandb login\n')

    def add_text(self, tag: str, text_string: Union[TextType], **kwargs):
        pass

    def add_scalar(self, tag: str, scalar_value: Union[ScalarType], **kwargs):
        pass


def get_dashboard(
        dashboard: Optional[str],
        project: str,
        logdir: str = "",
        config: Optional[Mapping] = None
) -> Union[Callable[[], WandBWriter], Callable[[], TensorboardWriter], Type[NilWriter]]:
    r"""Get the dashboard class.

    Args:
        dashboard
        project
        logdir
        config

    Examples:

        Tensorboard

        >>> TB = get_dashboard("tensorboard", project="filename", logdir="log/dir")
        >>> with TB() as writer:
        >>>     writer.add_scalar("tag", 1)
        >>>     writer.step()

    """
    if dashboard == "wandb":
        if wandb is not None:

            wandb.login()
            wandb.init(project=project)

            def _get_wandb():
                return WandBWriter(project)

            return _get_wandb
        else:
            dashboard = "tensorboard"

    if dashboard is None or dashboard == "tensorboard":

        def _get_tensorboard():
            return TensorboardWriter(logdir, project)

        return _get_tensorboard
    else:
        return NilWriter
