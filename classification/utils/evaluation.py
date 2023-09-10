import torch
from torch import nn
from timm.utils import accuracy

from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC,
)

# Typing imports
from torch.utils.data import DataLoader

from typing import Tuple


class MetricTracker(object):
    """This is a basic wrapper around torcheval's metric classes. This will be used to track the different metrics, and"""

    def __init__(self, num_classes: int, suffix: str = ""):
        self.suffix = suffix

        # Macro/Weighted metrics.
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.recall = MulticlassRecall(num_classes=num_classes)
        self.precision = MulticlassPrecision(num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)

        # Macro metrics
        self.macro_accuracy = MulticlassAccuracy(
            average="macro", num_classes=num_classes
        )
        self.macro_recall = MulticlassRecall(average="macro", num_classes=num_classes)
        self.macro_precision = MulticlassPrecision(
            average="macro", num_classes=num_classes
        )
        self.macro_f1_score = MulticlassF1Score(
            average="macro", num_classes=num_classes
        )
        self.macro_auc_roc = MulticlassAUROC(num_classes=num_classes, average="macro")

    def update(self, output: torch.Tensor, targets: torch.Tensor):
        # We do not need to cast the output per-class probabilities, as torcheval handles that internally

        # Macro/Weighted metrics.
        self.accuracy.update(output, targets)
        self.recall.update(output, targets)
        self.precision.update(output, targets)
        self.f1_score.update(output, targets)

        # Macro metrics
        self.macro_accuracy.update(output, targets)
        self.macro_recall.update(output, targets)
        self.macro_precision.update(output, targets)
        self.macro_f1_score.update(output, targets)
        self.macro_auc_roc.update(output, targets)

    def return_metrics(self) -> dict:
        return {
            f"accuracy{self.suffix}": self.accuracy.compute().item(),
            f"recall{self.suffix}": self.recall.compute().item(),
            f"precision{self.suffix}": self.precision.compute().item(),
            f"f1_score{self.suffix}": self.f1_score.compute().item(),
            f"macro_accuracy{self.suffix}": self.macro_accuracy.compute().item(),
            f"macro_recall{self.suffix}": self.macro_recall.compute().item(),
            f"macro_precision{self.suffix}": self.macro_precision.compute().item(),
            f"macro_f1_score{self.suffix}": self.macro_f1_score.compute().item(),
            f"macro_auc_roc{self.suffix}": self.macro_auc_roc.compute().item(),
        }


def evaluate(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    use_amp: bool = False,
    suffix: str = "",
) -> dict:
    """Evaluation step that returns a dict of metrics.

    Args:
        data_loader (DataLoader): The evaluation dataloader.
        model (nn.Module): The model used for evaluation. Can be the EMA-enabled model if specified.
        device (str): Device to use for model forward passes.
        use_amp (bool, optional): Whether to use AMP for evaluation. Defaults to False.
        suffix (str, optional): Whether to append a suffix to the metric dictionary keys. Defaults to a blank string.

    Returns:
        dict: Dictionary of computed metrics, will include [accuracy, recall, precision, f1_score, auc_roc, macro_accuracy, macro_recall, macro_precision, macro_f1_score].
    """
    # switch to evaluation mode
    model.eval()

    # Initialize the torcheval metric trackers.
    metric_tracker = MetricTracker(len(data_loader.dataset.class_to_idx), suffix=suffix)

    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(samples)
                if isinstance(output, dict):
                    output = output["logits"]

            metric_tracker.update(output, targets)

    return metric_tracker.return_metrics()
