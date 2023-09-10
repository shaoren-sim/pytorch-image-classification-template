import os
import csv
import torch
from contextlib import redirect_stdout

from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

# Typing imports
from yacs.config import CfgNode


def dump_yacs_as_yaml(cfg: CfgNode, dump_path: str):
    """Dumps the YACS CfgNode as a YAML file. Useful for reproducibility, as new experiments can inherit the exact settings dumped to the local file."""
    with open(dump_path, "w") as f:
        with redirect_stdout(f):
            print(cfg.dump())


def save_model_checkpoint(
    save_fp: str,
    epoch: int,
    model: nn.Module,
    model_ema: AveragedModel,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    best_metric: float,
    class_to_idx: dict,
):
    if model_ema is None:
        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_metric": best_metric,
                "class_to_idx": class_to_idx,
            },
            f=save_fp,
        )
    else:
        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": model_ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_metric": best_metric,
                "class_to_idx": class_to_idx,
            },
            f=save_fp,
        )


def csv_logger(filename, *args, reset=False):
    """Rudimentary CSV logger to log experiments."""
    if reset or not os.path.exists(filename):
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(args)
    else:
        with open(filename, "a+", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(args)


class CSVLogger(object):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def write(self, metric_dict: dict):
        # If the CSV does not exist yet, create it while initializing the headers.
        if not os.path.exists(self.filepath):
            csv_logger(self.filepath, *list(metric_dict.keys()))
        # Write the metrics to the CSV file.
        csv_logger(self.filepath, *list(metric_dict.values()))
