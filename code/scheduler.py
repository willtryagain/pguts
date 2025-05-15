import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    OneCycleLR,
    _LRScheduler,
)


class CycScheduler(OneCycleLR, _LRScheduler):
    def __init__(
        self, optimizer: Optimizer, max_lr: float, epochs: int, steps_per_epoch: int
    ):
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        super(CycScheduler, self).__init__(optimizer, max_lr, epochs, steps_per_epoch)


class CosineScheduler(CosineAnnealingLR, _LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        eta_min: float,
        T_max: int,
    ):
        self.eta_min = eta_min
        self.T_max = T_max
        super().__init__(optimizer, T_max, eta_min)


class CosineSchedulerWithRestarts(LambdaLR, _LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_factor: float = 0.1,
        linear_decay: float = 0.67,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ):
        """From https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/optimization.py#L138

        Create a schedule with a learning rate that decreases following the values
        of the cosine function between the initial lr set in the optimizer to 0,
        with several hard restarts, after a warmup period during which it increases
        linearly between 0 and the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`int`, *optional*, defaults to 1):
                The number of hard restarts to use.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.
        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                factor = float(current_step) / float(max(1, num_warmup_steps))
                return max(min_factor, factor)
            progress = float(current_step - num_warmup_steps)
            progress /= float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return 0.0
            factor = (float(num_cycles) * progress) % 1.0
            cos = 0.5 * (1.0 + math.cos(math.pi * factor))
            lin = 1.0 - (progress * linear_decay)
            return max(min_factor, cos * lin)

        super(CosineSchedulerWithRestarts, self).__init__(
            optimizer, lr_lambda, last_epoch
        )


if __name__ == "__main__":
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineSchedulerWithRestarts(
        optimizer=optimizer, num_warmup_steps=100, num_training_steps=1000
    )
    scheduler = CycScheduler(
        optimizer=optimizer, max_lr=1e-3, epochs=100, steps_per_epoch=100
    )
    print(scheduler.get_last_lr())
