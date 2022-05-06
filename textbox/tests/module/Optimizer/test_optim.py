from textbox.module.Optimizer import optim
import numpy as np
from matplotlib import pyplot as plt

init_lr, max_lr, n_warmup_steps = 0.01, 0.03, 20
max_steps = 200
steps = range(1, max_steps+1)


def plot(op):
    lrs = []
    op.n_steps = 0
    while op.n_steps < max_steps:
        op.n_steps += 1
        lrs.append(op.lr)

    assert len(steps) == len(lrs)

    plt.plot(steps, lrs)
    plt.title(type(op))
    plt.show()


def test_scheduled_lr():
    d_model = 100
    op = optim.ScheduledOptim(None, init_lr, d_model, n_warmup_steps)

    while op.n_steps < max_steps:
        op.n_steps += 1
        assert np.real_if_close(
            op.lr,
            (d_model ** -0.5) * min(op.n_steps ** (-0.5), op.n_steps * n_warmup_steps ** (-1.5))
        )

    plot(op)


def test_inverse_square_root_lr():
    op = optim.InverseSquareRootOptim(None, init_lr, max_lr, n_warmup_steps)
    plot(op)


def test_cosine_lr():
    op = optim.CosineOptim(None, init_lr, max_lr, n_warmup_steps, max_steps)
    plot(op)


def test_linear_lr():
    op = optim.LinearOptim(None, init_lr, max_lr, n_warmup_steps, max_steps)
    assert isinstance(op, optim.LinearOptim)
    plot(op)


def test_constant_lr():
    init_lr, max_lr, n_warmup_steps = 0.01, 0.03, 20
    op = optim.ConstantOptim(None, init_lr, max_lr, n_warmup_steps)
    assert isinstance(op, optim.ConstantOptim)
    plot(op)

