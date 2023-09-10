import os
import argparse

import timm
import torch
import torchvision
import numpy as np
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


# Typing-only imports, only used for typehints
from yacs.config import CfgNode
from typing import List


def get_parser() -> argparse.ArgumentParser:
    """Creates the parser to initialize an experiment.

    This should follow the YACS convention of 'There is only one way to configure the same thing.' Preferably, no additional CLI arguments should be added here. Instead, add them to the YACS configuration file, such that they can be overriden using the --config-overrides option.

    Returns:
        argparse.ArgumentParser: CLI Argument Parser.
    """
    parser = argparse.ArgumentParser(description="Image classification training.")
    parser.add_argument(
        "--cfg",
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
        help="Path to an existing checkpoint, will trigger the resume function.",
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

    # Merge overrides from the configuration file and command-line overrides.
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(cli_overrides)

    # Validate the config file settings through assertions defined in classification/config/initialization.py
    validate_cfg(cfg)

    # Freeze the config file to make it immutable.
    cfg.freeze()

    return cfg


def main(cfg: CfgNode, resume_ckpt_path: str = None):
    """Conducts an experiment based on the input experimental config.

    Args:
        cfg (CfgNode): Experimental configuration in the YACS format.
    """
    # Seeding for reproducibility
    torch.manual_seed(cfg.EXPERIMENT.SEED)
    np.random.seed(cfg.EXPERIMENT.SEED)

    # If active, turn on CUDNN benchmarking.
    if cfg.BACKEND.CUDNN_BENCHMARK:
        cudnn.benchmark = True

    # Dataset initialization.
    # Before, we need to initialize the transformations.
    transforms_train = build_train_transforms(
        image_size=cfg.MODEL.INPUT_SIZE,
        hflip_prob=cfg.AUGMENTATIONS.TRAIN.HFLIP_PROB,
        vflip_prob=cfg.AUGMENTATIONS.TRAIN.VFLIP_PROB,
        color_jitter_prob=cfg.AUGMENTATIONS.TRAIN.COLOR_JITTER_PROB,
        auto_augment_policy=cfg.AUGMENTATIONS.TRAIN.AUTO_AUGMENT_POLICY,
        interpolation_mode=cfg.AUGMENTATIONS.TRAIN.INTERPOLATION_MODE,
        random_erase_prob=cfg.AUGMENTATIONS.TRAIN.RANDOM_ERASE_PROB,
        random_erase_mode=cfg.AUGMENTATIONS.TRAIN.RANDOM_ERASE_MODE,
        random_erase_count=cfg.AUGMENTATIONS.TRAIN.RANDOM_ERASE_COUNT,
        pixel_mean=cfg.DATA.PIXEL_MEAN,
        pixel_std=cfg.DATA.PIXEL_STD,
        no_aug=cfg.AUGMENTATIONS.TRAIN.DISABLE,
    )
    transforms_eval = build_eval_transforms(
        image_size=cfg.MODEL.INPUT_SIZE,
        pixel_mean=cfg.DATA.PIXEL_MEAN,
        pixel_std=cfg.DATA.PIXEL_STD,
        crop_pct=cfg.AUGMENTATIONS.EVAL.CROP_PCT,
    )

    # Build the datasets.
    dataset_train, dataset_eval = build_dataset(
        cfg.DATA.DATASET,
        cfg.DATA.PRE_SPLIT_TRAIN_VAL,
        split_train=cfg.DATA.SPLIT_TRAIN,
        split_eval=cfg.DATA.SPLIT_EVAL,
        download_folder=cfg.BACKEND.DATASET_DOWNLOAD_FOLDER,
        transforms_train=transforms_train,
        transforms_eval=transforms_eval,
    )

    # We can also extract the class-to-index mapping
    class_to_idx = dataset_train.class_to_idx

    # Building the dataloaders.
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.DATALOADER.PER_STEP_BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.BACKEND.DATALOADER_PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    if dataset_eval is not None:
        dataloader_eval = DataLoader(
            dataset_train,
            batch_size=cfg.DATALOADER.PER_STEP_BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.BACKEND.DATALOADER_PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

    # Initializing model.
    model = build_model(
        model_name=cfg.MODEL.NAME,
        use_pretrained=cfg.MODEL.USE_PRETRAIN,
        num_classes=len(class_to_idx),
        dropout_rate=cfg.MODEL.DROPOUT_RATE,
        drop_path_rate=cfg.BAG_OF_TRICKS.STOCHASTIC_DEPTH.DROP_PATH_RATE,
    )

    # Bag of tricks - MixUp/CutMix
    if cfg.BAG_OF_TRICKS.MIXUP_CUTMIX.DO_CUTMIX:
        mixup_cutmix = build_mixup_cutmix(
            num_classes=len(class_to_idx),
            mixup_alpha=cfg.BAG_OF_TRICKS.MIXUP_CUTMIX.MIXUP_ALPHA,
            cutmix_alpha=cfg.BAG_OF_TRICKS.MIXUP_CUTMIX.CUTMIX_ALPHA,
            cutmix_minmax=cfg.BAG_OF_TRICKS.MIXUP_CUTMIX.CUTMIX_MINMAX,
            mixup_prob=cfg.BAG_OF_TRICKS.MIXUP_CUTMIX.MIXUP_PROB,
            switch_prob=cfg.BAG_OF_TRICKS.MIXUP_CUTMIX.SWITCH_PROB,
            mixup_mode=cfg.BAG_OF_TRICKS.MIXUP_CUTMIX.MIXUP_MODE,
            label_smoothing=cfg.LOSS_FUNCTION.LABEL_SMOOTHING,
        )
    else:
        mixup_cutmix = None

    # Build loss criterion
    criterion = build_loss_fn(
        mixup_cutmix_fn=mixup_cutmix,
        label_smoothing=cfg.LOSS_FUNCTION.LABEL_SMOOTHING,
    )

    # Deriving a few values, specifically the 'true' batch_size and the learning rate.
    # The 'true' batch size might be higher if gradient accumulation is used.
    # This calculates this apparent batch size, and will use it to scale down the learning rate.
    total_batch_size = cfg.DATALOADER.PER_STEP_BATCH_SIZE
    if cfg.DATALOADER.DO_GRADIENT_ACCUMULATION:
        print(
            "DATALOADER.DO_GRADIENT_ACCUMULATION is set to True. Calculating the apparent batch size."
        )
        total_batch_size = total_batch_size * cfg.DATALOADER.GRADIENT_ACCUMULATION_STEPS

    # As per most modern publications, the learning rate is scaled based on the batch size.
    # absolute_lr = base_lr * total_batch_size / 256
    learning_rate = cfg.OPTIMIZER.BASE_LEARNING_RATE
    if cfg.BACKEND.SCALE_LEARNING_RATE:
        print("Scaling learning rate based on batch size.")
        print(f"Apparent batch size = {total_batch_size}")
        learning_rate = learning_rate * total_batch_size / 256
        print(f"Scaled learning rate is {learning_rate}")

    training_steps = len(dataloader_train)

    # Building optimizer.
    optimizer = build_optimizer(
        param_groups=model.parameters(),
        optimizer_type=cfg.OPTIMIZER.TYPE,
        learning_rate=learning_rate,
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        momentum=cfg.OPTIMIZER.MOMENTUM,
        betas=cfg.OPTIMIZER.BETAS,
        eps=cfg.OPTIMIZER.EPS,
    )

    # Build learning rate scheduler.
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        lr_scheduler_type=cfg.LR_SCHEDULER.TYPE,
        total_steps=cfg.TRAINING.EPOCHS,
        warmup_steps=cfg.LR_SCHEDULER.WARMUP_STEPS,
        min_lr=cfg.LR_SCHEDULER.MIN_LR,
        update_period=cfg.LR_SCHEDULER.UPDATE_PERIOD,
        milestones=cfg.LR_SCHEDULER.MILESTONES,
        lr_multiplicative_factor=cfg.LR_SCHEDULER.LR_MULTIPLICATIVE_FACTOR,
    )

    # Default values, will be used if we are starting from scratch.
    start_epoch = 0
    if dataloader_eval is not None:
        best_metric = (
            -999999
        )  # All eval metrics follow higher=better convention, so start from the minimum metric.
    else:
        best_metric = 999999  # If no eval set is available, the training loss is used, which follows lower=better convention.

    # Send model to device.
    # This needs to happen before loading the state dictionaries.
    # There are some oddities when following the general convention of map_location="cpu".
    # See: https://discuss.pytorch.org/t/loading-a-model-runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least-two-devices-cuda-0-and-cpu/143897/9
    device = cfg.EXPERIMENT.DEVICE
    model.to(device)

    # Resume functionality if a --ckpt path is provided via the CLI.
    if resume_ckpt_path is not None:
        if not os.path.exists(resume_ckpt_path):
            print(
                f"Checkpoint {resume_ckpt_path} does not exist. Starting from scratch."
            )
        else:
            # Loading the checkpoint dict
            ckpt_dict = torch.load(resume_ckpt_path, map_location=device)

            # Loading the states of the model, optimizer and scheduler.
            model.load_state_dict(ckpt_dict["model_state_dict"], strict=True)
            optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
            lr_scheduler.load_state_dict(ckpt_dict["lr_scheduler_state_dict"])

            # Extracting the starting epoch and existing best metric. This will allow us to continue tracking the best existing model.
            start_epoch = (
                ckpt_dict["epoch"] + 1
            )  # Increment by 1, as the last epoch was already completed.
            best_metric = ckpt_dict["best_metric"]
            print(f"Resuming from checkpoint {resume_ckpt_path}.")
            print(
                f"Starting from epoch {start_epoch}, prior best metric: {best_metric:.4e}"
            )

    # Build the model EMA checkpointer.
    if cfg.BAG_OF_TRICKS.EMA.DO_EMA:
        model_ema = build_ema_averager(
            model=model, ema_decay=cfg.BAG_OF_TRICKS.EMA.EMA_DECAY
        )
        if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
            # Checkpoint dict was already loaded previously.
            model_ema.load_state_dict(ckpt_dict["ema_state_dict"])
    else:
        model_ema = None

    # AMP Scaler
    amp_device_type = "cuda" if "cuda" in cfg.EXPERIMENT.DEVICE else "cpu"
    use_amp = cfg.TRAINING.USE_AMP
    # AMP does not work on CPU, so force `use_amp` to be False if the CPU is selected.
    if "cuda" not in cfg.EXPERIMENT.DEVICE:
        print(f"CPU mode does not support AMP, so forcing use_amp to be False.")
        use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Everything is good to go.
    # Before starting the training loop, initializing the workdir and logging tools.
    workdir_path = cfg.EXPERIMENT.WORKDIR
    checkpoints_folder = os.path.join(workdir_path, "checkpoints")
    logs_folder = os.path.join(workdir_path, "logs")
    os.makedirs(workdir_path, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    # Initialize loggers.
    epoch_csv_path = os.path.join(logs_folder, "epoch_logs.csv")
    iter_csv_path = os.path.join(logs_folder, "iter_logs.csv")

    epoch_csv_logger = CSVLogger(epoch_csv_path)
    iter_csv_logger = CSVLogger(iter_csv_path)

    if cfg.LOGGING.DO_TENSORBOARD_LOGGING:
        tensorboard_writer = SummaryWriter(os.path.join(logs_folder, "tensorboard"))

    # Make a dump of the config file for reproducibility.
    dump_yacs_as_yaml(cfg, os.path.join(workdir_path, "config.yaml"))

    # Main training loop
    for epoch in range(start_epoch, cfg.TRAINING.EPOCHS):
        # Track losses for each epoch.
        moving_loss = []

        # Training phase.
        model.train()
        for data_iter_step, (samples, targets) in enumerate(
            dataloader_train, start=epoch * training_steps
        ):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_cutmix is not None:
                samples, targets = mixup_cutmix(samples, targets)

            with torch.autocast(
                device_type=amp_device_type,
                enabled=use_amp,
            ):
                output = model(samples)
                loss = criterion(output, targets)

            # Optional flag that allows the code to break if NaN loss is achieved.
            if cfg.BACKEND.BREAK_WHEN_LOSS_NAN and torch.isnan(loss):
                raise RuntimeError(
                    f"Loss became NaN at step {data_iter_step}, epoch {epoch}."
                )

            if cfg.DATALOADER.DO_GRADIENT_ACCUMULATION:
                optimize_gradient_accumulation(
                    loss=loss,
                    step=data_iter_step % training_steps,
                    sub_batches=cfg.DATALOADER.GRADIENT_ACCUMULATION_STEPS,
                    epoch_length=training_steps,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    do_gradient_clipping=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.DO_GRADIENT_CLIPPING,
                    gradient_clipping_method=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.METHOD,
                    gradient_clipping_norm_type=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.NORM_MAX,
                    gradient_clipping_norm_max=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.NORM_MAX,
                    gradient_clipping_value=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.VALUE,
                    do_amp=use_amp,
                )
            else:
                optimize(
                    loss=loss,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    do_gradient_clipping=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.DO_GRADIENT_CLIPPING,
                    gradient_clipping_method=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.METHOD,
                    gradient_clipping_norm_type=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.NORM_MAX,
                    gradient_clipping_norm_max=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.NORM_MAX,
                    gradient_clipping_value=cfg.BAG_OF_TRICKS.GRADIENT_CLIPPING.VALUE,
                    do_amp=use_amp,
                )

            # Update the EMA model if available.
            if not cfg.DATALOADER.DO_GRADIENT_ACCUMULATION:
                # In the case of there being no gradient accumulation, its basic enough.
                # Just update at the right interval.
                if model_ema is not None and (
                    (data_iter_step + 1) % cfg.BAG_OF_TRICKS.EMA.EMA_UPDATE_INTERVAL
                    == 0
                ):
                    print(f"Updating EMA model parameters at step {data_iter_step + 1}")
                    model_ema.update_parameters(model)
            else:
                # However, with gradient accumulation, the right number of actual updates needs to have elapsed.
                # This has to be calculated.
                # Since this implementation of gradient accumulation still updates at the end of an epoch.
                # We need to calculate the number of grad accumum steps per epoch.
                grad_accum_steps_per_epoch = int(
                    np.ceil(training_steps / cfg.DATALOADER.GRADIENT_ACCUMULATION_STEPS)
                )
                current_elapsed_grad_accum_steps = (
                    epoch + 1
                ) * grad_accum_steps_per_epoch + (
                    data_iter_step % training_steps
                ) / cfg.DATALOADER.GRADIENT_ACCUMULATION_STEPS

                if model_ema is not None and (
                    current_elapsed_grad_accum_steps
                    % cfg.BAG_OF_TRICKS.EMA.EMA_UPDATE_INTERVAL
                    == 0
                ):
                    print(
                        f"Updating EMA model parameters at step {data_iter_step + 1}, after {current_elapsed_grad_accum_steps // cfg.BAG_OF_TRICKS.EMA.EMA_UPDATE_INTERVAL} gradient accumulation steps."
                    )
                    model_ema.update_parameters(model)

            training_loss = loss.detach().cpu().item()
            moving_loss.append(training_loss)

            # Log per-iteration
            print(
                f"Epoch {epoch}: {data_iter_step % training_steps + 1}/{training_steps}, train_loss: {training_loss:.4f}"
            )
            if data_iter_step % cfg.LOGGING.LOGGING_ITER_INTERVAL == 0:
                iter_step_dict = {
                    "step": data_iter_step,
                    "training_loss_iter": training_loss,
                }

                iter_csv_logger.write(iter_step_dict)
                if cfg.LOGGING.DO_TENSORBOARD_LOGGING:
                    # Pop the "iter_step" key, since Tensorboard logs this as the 'global_step' parameter.
                    iter_step_dict.pop("step")

                    for key, value in iter_step_dict.items():
                        tensorboard_writer.add_scalar(key, value, data_iter_step)

        # Evaluation section.
        if dataloader_eval is not None:
            eval_metrics_dict = evaluate(
                data_loader=dataloader_eval,
                model=model,
                device=cfg.EXPERIMENT.DEVICE,
                use_amp=use_amp,
            )

            # Run evaluation on the EMA model as well if requested.
            if cfg.EVALUATION.TRACK_EMA:
                # Evaluate the
                eval_metrics_ema_dict = evaluate(
                    data_loader=dataloader_eval,
                    model=model_ema,
                    device=cfg.EXPERIMENT.DEVICE,
                    use_amp=use_amp,
                    suffix="_ema",
                )

            # Check if the model has improved
            if not cfg.EVALUATION.SELECT_EMA_MODEL:
                # Non EMA tracking.
                monitored_metric = eval_metrics_dict[f"{cfg.EVALUATION.METRIC}"]
                if monitored_metric > best_metric:
                    print(
                        f"Best metric {cfg.EVALUATION.METRIC} improved from {best_metric:.4e} to {monitored_metric:.4e}"
                    )
                    best_metric = monitored_metric
                    _save_model_path = os.path.join(checkpoints_folder, "best.ckpt")
                    print(f"Saving new best model to {_save_model_path}")
                    save_model_checkpoint(
                        _save_model_path,
                        epoch,
                        model,
                        model_ema,
                        optimizer,
                        lr_scheduler,
                        best_metric,
                        class_to_idx,
                    )
            else:
                # EMA tracking.
                monitored_metric = eval_metrics_ema_dict[f"{cfg.EVALUATION.METRIC}_ema"]
                if monitored_metric > best_metric:
                    print(
                        f"Best metric {cfg.EVALUATION.METRIC}_ema  improved from {best_metric:.4e} to {monitored_metric:.4e}"
                    )
                    best_metric = monitored_metric
                    _save_model_path = os.path.join(checkpoints_folder, "best.ckpt")
                    print(f"Saving new best model to {_save_model_path}")
                    save_model_checkpoint(
                        _save_model_path,
                        epoch,
                        model,
                        model_ema,
                        optimizer,
                        lr_scheduler,
                        best_metric,
                        class_to_idx,
                    )
        else:
            # This case is triggered when there is no evaluation set. In this case, the training loss is used for model selection.
            monitored_metric = (
                training_loss  # This assignment is primarily for the logging section.
            )
            if training_loss < best_metric:
                print(
                    f"Best training loss improved from {best_metric:.4e} to {training_loss:.4e}"
                )
                best_metric = training_loss
                _save_model_path = os.path.join(checkpoints_folder, "best_model.ckpt")
                print(f"Saving new best model to {_save_model_path}")
                save_model_checkpoint(
                    _save_model_path,
                    epoch,
                    model,
                    model_ema,
                    optimizer,
                    lr_scheduler,
                    best_metric,
                    class_to_idx,
                )

        # After every epoch, save a checkpoint
        save_model_checkpoint(
            os.path.join(checkpoints_folder, "last.ckpt"),
            epoch,
            model,
            model_ema,
            optimizer,
            lr_scheduler,
            best_metric,
            class_to_idx,
        )

        # Log all metrics.
        # Calculating the epoch loss.
        epoch_training_loss = np.array(moving_loss)
        epoch_training_loss = epoch_training_loss[~np.isnan(epoch_training_loss)]
        epoch_training_loss = np.mean(epoch_training_loss)

        # For the default CSV logging, construct the metric dict.
        all_metrics_dict = {
            "epoch": epoch,
            "learning_rate": lr_scheduler.get_last_lr()[0],
            "training_loss_epoch": epoch_training_loss,
            "model_selection_metric": monitored_metric,
        }
        # Merging the eval dictionaries if they exist.
        if dataloader_eval is not None:
            all_metrics_dict = dict(all_metrics_dict, **eval_metrics_dict)
            if cfg.EVALUATION.TRACK_EMA:
                all_metrics_dict = dict(all_metrics_dict, **eval_metrics_ema_dict)
        print(all_metrics_dict)
        epoch_csv_logger.write(all_metrics_dict)

        if cfg.LOGGING.DO_TENSORBOARD_LOGGING:
            # Pop the "epoch" key, since Tensorboard logs this as the 'global_step' parameter.
            all_metrics_dict.pop("epoch")

            for key, value in all_metrics_dict.items():
                tensorboard_writer.add_scalar(key, value, epoch)

        # After an epoch has passed, advance the lr_scheduler.
        lr_scheduler.step()


if __name__ == "__main__":
    # Parse arguments from the CLI.
    args = get_parser().parse_args()
    config_file = args.cfg
    cli_overrides = args.overrides
    existing_ckpt = args.ckpt

    # Generates the experimental configuration node.
    cfg = setup(config_path=config_file, cli_overrides=cli_overrides)

    # Conduct the training loop.
    main(cfg, existing_ckpt)
