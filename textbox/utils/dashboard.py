"""Provides several APIs for dashboard including :py:mod:`torch.utils.tensorboard`

Todo:
    * Implement WandBWriter
"""

import os
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger

from typing import Optional, Any, Type, Union, Iterable
ScalarType = (float, int)
TextType = str


class AbstractDashboard:

    def __init__(self, logdir: str):
        self.global_step: int = 0
        self.logdir = logdir

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
        if isinstance(any_value, Iterable):
            self.add_iterable(tag, any_value, **kwargs)
        elif isinstance(any_value, dict):
            self.add_dict(tag, any_value, **kwargs)
        elif isinstance(any_value, ScalarType):
            self.add_scalar(tag, any_value, **kwargs)


class NilWriter(AbstractDashboard):

    def __init__(self, logdir: str = ""):
        super().__init__(logdir)

    def add_scalar(self, tag: str, scalar_value: Any, **kwargs):
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
    def __init__(self, logdir: str, comment: str = "_tensorboard"):
        logdir = os.path.join(logdir, "tensorboard")
        super(TensorboardWriter, self).__init__(logdir)
        self.dashboard = SummaryWriter(log_dir=logdir, comment=comment)

    def __enter__(self) -> "TensorboardWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dashboard.flush()
        self.dashboard.close()

    def add_scalar(self, tag, scalar_value, **kwargs):
        self.dashboard.add_scalar(tag, scalar_value, global_step=self.global_step, **kwargs)

    def add_text(self, tag, text_string, **kwargs):
        self.dashboard.add_text(tag, text_string, global_step=self.global_step, **kwargs)


def get_dashboard(dashboard: Optional[str]) -> Type[AbstractDashboard]:
    r"""Get the dashboard class."""
    if dashboard is None or dashboard == "tensorboard":
        return TensorboardWriter
    else:
        return NilWriter
