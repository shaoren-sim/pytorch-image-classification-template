import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# Typing imports
from timm.data import Mixup
from torch.nn.modules.loss import _Loss


def build_loss_fn(mixup_cutmix_fn: Mixup, label_smoothing: float) -> _Loss:
    """This function comes from the ConvNeXt-V2 codebase, and builds CrossEntropyLoss variants depending on the presence of mixup or cutmix, or label smoothing.

    Args:
        mixup_cutmix_fn (Mixup): Mixup/CutMix function, could be None.
        label_smoothing (float): Label smoothing amount, if larger than 0.0, LabelSmoothingCrossEntropy will be used instead.

    Returns:
        _Loss: CrossEntropyLoss variations.
    """
    if mixup_cutmix_fn is not None:
        # smoothing is handled with mixup label transform
        return SoftTargetCrossEntropy()
    else:
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
