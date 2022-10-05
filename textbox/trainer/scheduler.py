# @Time   : 2020/11/14
# @Author : Junyi Li
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2022/05/06
# @Author : Hu Yiwen
# @Email  : huyiwen@ruc.edu.cn

r"""
Optimizer
#####################
"""

import numpy as np

from torch.optim import Optimizer

from typing import Tuple

SchedulerState = Tuple[dict, int]


class AbstractScheduler:
    r"""Abstract Base Class of Scheduler. To modify your new scheduler, simply inherit
    this class and implement `__init__()` and `lr()` with ``@property`` decorator.
    """

    def __init__(self, base_optimizer: Optimizer, init_lr: float):
        self.optimizer = base_optimizer
        self.init_lr = init_lr
        self.n_steps: int = 0

    def step(self):
        self._update_learning_rate()
        self.optimizer.step()

    def _update_learning_rate(self):
        r"""Update learning rate. One just need to implement `lr` property."""
        self.n_steps += 1
        lr = self.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def lr(self):
        r"""Get learning rate for current step."""
        raise NotImplementedError

    def state_dict(self) -> SchedulerState:
        return self.optimizer.state_dict(), self.n_steps

    def load_state_dict(self, state_dict: SchedulerState):
        opt, self.n_steps = state_dict
        self.optimizer.load_state_dict(opt)

    def __getattr__(self, item: str):
        r"""Pass method calls e.g. `zero_grad()`.
        One can override these methods by simply implementing new methods.
        """
        return getattr(self.optimizer, item)


class InverseSquareRootScheduler(AbstractScheduler):

    def __init__(self, base_optimizer: Optimizer, init_lr: float, max_lr: float, n_warmup_steps: int):
        super().__init__(base_optimizer, init_lr)
        self.n_warmup_steps = n_warmup_steps

        self.warmup_k = (max_lr - init_lr) / n_warmup_steps
        self.decay_k = max_lr * (n_warmup_steps ** 0.5)

    @property
    def lr(self):
        if self.n_steps <= self.n_warmup_steps:
            return self.init_lr + self.warmup_k * self.n_steps
        else:
            return self.decay_k * self.n_steps ** -0.5


class CosineScheduler(AbstractScheduler):

    def __init__(self, base_optimizer: Optimizer, init_lr: float, max_lr: float, n_warmup_steps: int, max_steps: int):
        super().__init__(base_optimizer, init_lr)
        self.n_warmup_steps = n_warmup_steps
        self.half_delta = (max_lr - init_lr) / 2

        self.warmup_k = (max_lr - init_lr) / n_warmup_steps
        self.decay_k = np.pi / (max_steps - n_warmup_steps)

    @property
    def lr(self):
        if self.n_steps <= self.n_warmup_steps:
            return self.init_lr + self.warmup_k * self.n_steps
        else:
            return self.init_lr + self.half_delta * (1. + np.cos(self.decay_k * (self.n_steps - self.n_warmup_steps)))


class LinearScheduler(AbstractScheduler):

    def __init__(self, base_optimizer: Optimizer, init_lr: float, max_lr: float, n_warmup_steps: int, max_steps: int):
        super().__init__(base_optimizer, init_lr)
        self.n_warmup_steps = n_warmup_steps
        self.init_lr = init_lr
        self.max_lr = max_lr

        self.warmup_k = (max_lr - init_lr) / n_warmup_steps
        self.decay_k = (max_lr - init_lr) / (max_steps - n_warmup_steps)  # decay to zero

    @property
    def lr(self):
        if self.n_steps <= self.n_warmup_steps:
            return self.init_lr + self.warmup_k * self.n_steps
        else:
            return self.max_lr - self.decay_k * (self.n_steps - self.n_warmup_steps)


class ConstantScheduler(AbstractScheduler):

    def __init__(self, base_optimizer: Optimizer, init_lr: float, max_lr: float, n_warmup_steps: int):
        super().__init__(base_optimizer, init_lr)
        self.n_warmup_steps = n_warmup_steps
        self.max_lr = max_lr

        self.warmup_k = (max_lr - init_lr) / n_warmup_steps

    @property
    def lr(self):
        if self.n_steps <= self.n_warmup_steps:
            return self.init_lr + self.warmup_k * self.n_steps
        else:
            return self.max_lr
