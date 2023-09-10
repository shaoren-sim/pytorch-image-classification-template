import torch
from torch import nn
from timm.utils import accuracy

# Typing imports
from torch.optim.optimizer import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from typing import Tuple


def optimize(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    do_gradient_clipping: bool = False,
    gradient_clipping_method: str = "norm",
    gradient_clipping_norm_max: float = 5.0,
    gradient_clipping_norm_type: int = 2,
    gradient_clipping_value: float = 1e-4,
    do_amp: bool = False,
):
    # Backpropagate.
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    # See https://github.com/pytorch/pytorch/issues/309#issuecomment-327304962 for the proper implementation of gradient clipping.
    if do_gradient_clipping:
        # Gradient needs to be unscaled.
        scaler.unscale_(optimizer)
        if gradient_clipping_method == "norm":
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clipping_norm_max,
                norm_type=gradient_clipping_norm_type,
                error_if_nonfinite=not do_amp,
            )
        elif gradient_clipping_method == "value":
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                clip_value=gradient_clipping_value,
            )
    scaler.step(optimizer)
    scaler.update()


def optimize_gradient_accumulation(
    loss: torch.Tensor,
    step: int,
    sub_batches: int,
    epoch_length: int,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    do_gradient_clipping: bool = False,
    gradient_clipping_method: str = "norm",
    gradient_clipping_norm_max: float = 5.0,
    gradient_clipping_norm_type: int = 2,
    gradient_clipping_value: float = 1e-4,
    do_amp: bool = False,
):
    # Backpropagate.
    scaler.scale(loss).backward()

    # Only update after X steps are elapsed.
    if ((step + 1) % sub_batches == 0) or (step + 1 >= epoch_length):
        print(f"Updating parameters with accumulated gradients at step {step+1}")
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # See https://github.com/pytorch/pytorch/issues/309#issuecomment-327304962 for the proper implementation of gradient clipping.
        if do_gradient_clipping:
            # Gradient needs to be unscaled.
            scaler.unscale_(optimizer)
            if gradient_clipping_method == "norm":
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clipping_norm_max,
                    norm_type=gradient_clipping_norm_type,
                    error_if_nonfinite=not do_amp,
                )
            elif gradient_clipping_method == "value":
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    clip_value=gradient_clipping_value,
                )
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()
