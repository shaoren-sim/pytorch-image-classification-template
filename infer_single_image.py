import os
import argparse

import timm
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.utils import ModelEma

from classification.utils.tracking import (
    dump_yacs_as_yaml,
    save_model_checkpoint,
    CSVLogger,
)
from classification.config.initialization import get_default_cfg, validate_cfg
from classification.builders.transforms import (
    build_train_transforms,
    build_eval_transforms,
)
from classification.builders.dataset import build_dataset
from classification.builders.bag_of_tricks import build_mixup_cutmix, build_ema_averager
from classification.builders.model import build_model
from classification.builders.loss import build_loss_fn
from classification.builders.optimizer import build_optimizer
from classification.builders.scheduler import build_lr_scheduler

from classification.utils.training import optimize, optimize_gradient_accumulation
from classification.utils.evaluation import evaluate

from PIL import Image


# Typing-only imports, only used for typehints
from yacs.config import CfgNode
from typing import List


def get_parser() -> argparse.ArgumentParser:
    """Creates the parser to initialize an inference run.

    This should follow the YACS convention of 'There is only one way to configure the same thing.' Preferably, no additional CLI arguments should be added here. Instead, add them to the YACS configuration file, such that they can be overriden using the --config-overrides option.

    Returns:
        argparse.ArgumentParser: CLI Argument Parser.
    """
    parser = argparse.ArgumentParser(description="Image classification inference.")
    parser.add_argument(
        "--config-file",
        required=True,
        metavar="FILE",
        help="Path to YACS config file in .yaml format.",
    )
    parser.add_argument(
        "--overrides",
        metavar="STRING",
        default=[],
        type=str,
        help=(
            "Modify experimental configuration from the command line. See https://github.com/rbgirshick/yacs#command-line-overrides for details. Inputs should be comma-separated: 'python train.py --config-overrides EXPERIMENT.NAME modified_exp MODEL.NAME timm_resnet_18'."
        ),
        nargs=argparse.ONE_OR_MORE,
    )
    parser.add_argument(
        "--ckpt",
        required=False,
        default=None,
        metavar="FILE",
        help="Path to an existing checkpoint, if not provided, will load the best.ckpt file from the workdir.",
    )
    parser.add_argument(
        "--img",
        required=True,
        default=None,
        metavar="FILE",
        help="Path to an image.",
    )
    parser.add_argument(
        "--device",
        required=False,
        default="cpu",
        help="Device to use for inference.",
    )

    return parser


def setup(config_path: str, cli_overrides: List[str]) -> CfgNode:
    """Initialize the experimental configuration, and return an immutable configuration as per the YACS convention.

    Args:
        config_path (str): Path to the YACS config file.
        cli_overrides (List[str]): CLI overrides for experimental parameters. Should be a list of

    Returns:
        CfgNode: _description_
    """
    cfg = get_default_cfg()
    dump_yacs_as_yaml(cfg, "./config.yaml")

    # Merge overrides from the configuration file and command-line overrides.
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(cli_overrides)

    # Validate the config file settings through assertions defined in classification/config/initialization.py
    validate_cfg(cfg)

    # Freeze the config file to make it immutable.
    cfg.freeze()

    return cfg


def main(cfg: CfgNode, ckpt_path: str, img_path: str, device: str):
    """Starts an inference run on a single image.

    Args:
        cfg (CfgNode): Experimental configuration in the YACS format.
        ckpt_path (str): The checkpoint to load weights from.
        img_path (str): Path to the image to run inference on.
    """
    # If active, turn on CUDNN benchmarking.
    if cfg.BACKEND.CUDNN_BENCHMARK:
        cudnn.benchmark = True

    # Loading the checkpoint.
    # This needs to happen early, as the class_to_idx mapping is saved in the checkpoint file.
    if ckpt_path is None:
        # If no ckpt_path is provided, default to using the best.ckpt checkpoint.
        # This does assume that the workdir was not moved though.
        ckpt_path = os.path.join(cfg.EXPERIMENT.WORKDIR, "checkpoints", "best.ckpt")
    ckpt_dict = torch.load(ckpt_path, map_location="cpu")

    # We can also extract the class-to-index mapping
    class_to_idx = ckpt_dict["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Building the model.
    model = build_model(
        model_name=cfg.MODEL.NAME,
        use_pretrained=cfg.MODEL.USE_PRETRAIN,
        num_classes=len(class_to_idx),
        dropout_rate=cfg.MODEL.DROPOUT_RATE,
        drop_path_rate=cfg.BAG_OF_TRICKS.STOCHASTIC_DEPTH.DROP_PATH_RATE,
    )

    if cfg.EVALUATION.SELECT_EMA_MODEL:
        # Build the EMA model.
        model = build_ema_averager(model, ema_decay=0.0)
        model.load_state_dict(ckpt_dict["ema_state_dict"], strict=True)
        print(f"Using EMA model.")
    else:
        # Otherwise, just load the standard model.
        model.load_state_dict(ckpt_dict["model_state_dict"], strict=True)
    model.to(device)

    transform = build_eval_transforms(
        image_size=cfg.MODEL.INPUT_SIZE,
        pixel_mean=cfg.DATA.PIXEL_MEAN,
        pixel_std=cfg.DATA.PIXEL_STD,
        crop_pct=cfg.AUGMENTATIONS.EVAL.CROP_PCT,
    )

    # The model can be set to eval mode.
    model.eval()

    # Load the image.
    img = Image.open(img_path).convert("RGB")

    with torch.no_grad():
        model.eval()
        img_tensor = transform(img)
        output = model(img_tensor.unsqueeze(0))
        confidence = torch.nn.functional.softmax(output, dim=1)
        index = output.cpu().numpy().argmax()
        class_name = idx_to_class[index]
        print(
            f"Class is '{class_name}' with confidence of {confidence[0][index]*100:.4f}%"
        )

        # Printing the per-class confidences.
        for ind, conf in enumerate(confidence[0]):
            print(f"{idx_to_class[ind]:<15}: {conf*100:.4f}%")


if __name__ == "__main__":
    # Parse arguments from the CLI.
    args = get_parser().parse_args()
    config_file = args.config_file
    cli_overrides = args.overrides
    ckpt = args.ckpt
    img = args.img
    device = args.device

    # Generates the experimental configuration node.
    cfg = setup(config_path=config_file, cli_overrides=cli_overrides)

    # Conduct the training loop.
    main(cfg, ckpt, img, device)
