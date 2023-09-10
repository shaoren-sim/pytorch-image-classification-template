"""This file contains a few builders for methods that might qualify as bag-of-tricks. For example, there is MixUp and CutMix, as well as other potential methods that might be considered in the future."""

from torch import nn
from torch.optim.swa_utils import AveragedModel
from timm.data.mixup import Mixup

# Typing imports
from typing import Union


def build_mixup_cutmix(
    num_classes: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    cutmix_minmax: float = None,
    mixup_prob: float = 1.0,
    switch_prob: float = 0.5,
    mixup_mode: str = "batch",
    label_smoothing: float = 0.1,
) -> Union[Mixup, None]:
    """_summary_

    Args:
        num_classes (int): Number of image classes. Likely obtained via the Dataset's .class_to_idx property.
        mixup_alpha (float, optional): Alpha parameter for Mixup. Mixup is not active if mixup_alpha = 0.0. Defaults to 0.0.
        cutmix_alpha (float, optional): Alpha parameter for CutMix. CutMix is not active if cutmix_alpha = 0.0. Defaults to 0.0.
        cutmix_minmax (float, optional): CutMix min/max ratio, overrides alpha and enables CutMix if set (default: None). Defaults to None.
        mixup_prob (float, optional): Probability of performing mixup or cutmix when either/both is enabled. Defaults to 1.0.
        switch_prob (float, optional): Probability of switching to cutmix when both mixup and cutmix enabled. Defaults to 0.5.
        mixup_mode (str, optional): How to apply mixup/cutmix params. Per "batch", "pair", or "elem". Defaults to "batch".
        label_smoothing (float, optional): Label smoothing parameter. Defaults to 0.1.

    Returns:
        Union[Mixup, None]: If either Mixup or CutMix is enabled, returns the timm.data.Mixup object. Otherwise, returns None.
    """
    mixup_fn = None
    mixup_active = mixup_alpha > 0 or cutmix_alpha > 0.0 or cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=cutmix_minmax,
            prob=mixup_prob,
            switch_prob=switch_prob,
            mode=mixup_mode,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )

    return mixup_fn


def build_ema_averager(model: nn.Module, ema_decay: float) -> AveragedModel:
    """Function to create an EMA model.

    This is a modification of the original codebase, which uses the ModelEMA function from timm, which is marked as DEPRECATED.

    Args:
        model (nn.Module): The classification model.
        ema_decay (float): EMA decay factor. Should be between [0.0, 1.0]. For example, if set to 0.99, the AveragedModel will be updated by the 0.99*model_parameters.

    Returns:
        AveragedModel: The EMA model.
    """
    # See https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies for native PyTorch implementation
    ema_avg_fn = (
        lambda averaged_model_parameter, model_parameter, num_averaged: ema_decay
        * averaged_model_parameter
        + (1.0 - ema_decay) * model_parameter
    )
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn)

    return ema_model
