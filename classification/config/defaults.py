from yacs.config import CfgNode as CN

_C = CN()

# Generatl experiment settings.
_C.EXPERIMENT = CN()
_C.EXPERIMENT.WORKDIR = "./workdir"
_C.EXPERIMENT.DEVICE = "cuda:0"
_C.EXPERIMENT.SEED = 3407

# Model settings.
_C.MODEL = CN()
_C.MODEL.NAME = "timm_convnext_atto"
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.INPUT_SIZE = 224
_C.MODEL.DROPOUT_RATE = 0.0

# Dataset options.
_C.DATA = CN()
# Dataset can be a path to a local folder. This will use it as a torchvision.datasets.ImageFolder dataset.
# Alternatively, use a torchvision dataset using torchvision_{dataset_name}.
_C.DATA.DATASET = "path/to/image/folder"  # or torchvision_cifar10
_C.DATA.PRE_SPLIT_TRAIN_VAL = False  # If True, assumes that the folder has a 'train' and 'val' folder, which will be used.
# Split settings are only used when necessary, such as when using ImageFolder datasets.
_C.DATA.SPLIT_TRAIN = 0.8
_C.DATA.SPLIT_EVAL = 0.2
# Data normalization settings.
# These are the Imagenet values.
# _C.DATA.PIXEL_MEAN = [0.485, 0.456, 0.406]
# _C.DATA.PIXEL_STD = [0.229, 0.224, 0.225]
_C.DATA.PIXEL_MEAN = [0.0, 0.0, 0.0]
_C.DATA.PIXEL_STD = [1.0, 1.0, 1.0]

# Data augmentations
_C.AUGMENTATIONS = CN()
# Training augmentations are adapted from the ConvNeXt-V2 repo (https://github.com/facebookresearch/ConvNeXt-V2/blob/main/datasets.py#L50), with some components deactivated like the AutoAugment policies.
_C.AUGMENTATIONS.TRAIN = CN()
_C.AUGMENTATIONS.TRAIN.DISABLE = (
    False  # Debug flag to disable the training augmentations.
)
_C.AUGMENTATIONS.TRAIN.HFLIP_PROB = 0.5
_C.AUGMENTATIONS.TRAIN.VFLIP_PROB = 0.0
_C.AUGMENTATIONS.TRAIN.COLOR_JITTER_PROB = 0.4
_C.AUGMENTATIONS.TRAIN.AUTO_AUGMENT_POLICY = None
_C.AUGMENTATIONS.TRAIN.INTERPOLATION_MODE = "bilinear"
_C.AUGMENTATIONS.TRAIN.RANDOM_ERASE_PROB = 0.0
_C.AUGMENTATIONS.TRAIN.RANDOM_ERASE_MODE = "const"
_C.AUGMENTATIONS.TRAIN.RANDOM_ERASE_COUNT = 1
# Evaluation augmentations are also primarily from ConvNeXt-V2.
_C.AUGMENTATIONS.EVAL = CN()
_C.AUGMENTATIONS.EVAL.CROP_PCT = None

# Dataloader specific settings.
_C.DATALOADER = CN()
_C.DATALOADER.PER_STEP_BATCH_SIZE = 64
_C.DATALOADER.NUM_WORKERS = 4
# Gradient accumulation allows for larger batch sizes to be simulated.
# This only happens when DO_GRADIENT_ACCUMULATION=Truie.
# The simulated batch sizes depend on the number of GRADIENT_ACCUMULATION_STEPS.
# If PER_STEP_BATCH_SIZE=32 and GRADIENT_ACCUMULATION_STEPS=32, the apparent batch size will be PER_STEP_BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS=1024
_C.DATALOADER.DO_GRADIENT_ACCUMULATION = False
_C.DATALOADER.GRADIENT_ACCUMULATION_STEPS = 32

# Loss criterion settings.
_C.LOSS_FUNCTION = CN()
_C.LOSS_FUNCTION.LABEL_SMOOTHING = 0.1

# Optimizer settings.
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = "AdamW"
_C.OPTIMIZER.BASE_LEARNING_RATE = 5e-4
_C.OPTIMIZER.WEIGHT_DECAY = 0.05
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.BETAS = (0.9, 0.999)
_C.OPTIMIZER.EPS = 1e-8

# Learning rate scheduler settings.
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYPE = "CosineAnnealingWithWarmup"
_C.LR_SCHEDULER.WARMUP_STEPS = 20
_C.LR_SCHEDULER.MIN_LR = 1e-6
# The following settings only apply to step-based or multiplicative learning rates, not the cosine schedulers we useby default.
_C.LR_SCHEDULER.UPDATE_PERIOD = 5
_C.LR_SCHEDULER.MILESTONES = [60, 80]
_C.LR_SCHEDULER.LR_MULTIPLICATIVE_FACTOR = 0.95

_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 100
_C.TRAINING.USE_AMP = True

_C.EVALUATION = CN()
_C.EVALUATION.METRIC = "accuracy"  # Best metric to keep track of.
_C.EVALUATION.TRACK_EMA = True  # If True, evaluation will be run on the EMA model.
_C.EVALUATION.SELECT_EMA_MODEL = False  # If True, will save the best EMA model.

# Bag Of Tricks
_C.BAG_OF_TRICKS = CN()

# MixUp (https://arxiv.org/abs/1710.09412)
# CutMix (https://arxiv.org/abs/1905.04899)
_C.BAG_OF_TRICKS.MIXUP_CUTMIX = CN()
_C.BAG_OF_TRICKS.MIXUP_CUTMIX.DO_CUTMIX = True
_C.BAG_OF_TRICKS.MIXUP_CUTMIX.MIXUP_ALPHA = 0.1
_C.BAG_OF_TRICKS.MIXUP_CUTMIX.CUTMIX_ALPHA = 0.1
_C.BAG_OF_TRICKS.MIXUP_CUTMIX.CUTMIX_MINMAX = None
_C.BAG_OF_TRICKS.MIXUP_CUTMIX.MIXUP_PROB = 1.0
_C.BAG_OF_TRICKS.MIXUP_CUTMIX.SWITCH_PROB = 0.5
_C.BAG_OF_TRICKS.MIXUP_CUTMIX.MIXUP_MODE = "batch"

# Stochastic Depth (https://github.com/huggingface/pytorch-image-models/blob/a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models/layers/drop.py#L140)
_C.BAG_OF_TRICKS.STOCHASTIC_DEPTH = CN()
_C.BAG_OF_TRICKS.STOCHASTIC_DEPTH.DROP_PATH_RATE = 0.1

# Exponential Moving Average (EMA)
_C.BAG_OF_TRICKS.EMA = CN()
_C.BAG_OF_TRICKS.EMA.DO_EMA = True
_C.BAG_OF_TRICKS.EMA.EMA_DECAY = 0.99
_C.BAG_OF_TRICKS.EMA.EMA_UPDATE_INTERVAL = 10

# Gradient clipping
_C.BAG_OF_TRICKS.GRADIENT_CLIPPING = CN()
_C.BAG_OF_TRICKS.GRADIENT_CLIPPING.DO_GRADIENT_CLIPPING = False
_C.BAG_OF_TRICKS.GRADIENT_CLIPPING.METHOD = "norm"
_C.BAG_OF_TRICKS.GRADIENT_CLIPPING.NORM_TYPE = 2
_C.BAG_OF_TRICKS.GRADIENT_CLIPPING.NORM_MAX = 5.0
_C.BAG_OF_TRICKS.GRADIENT_CLIPPING.VALUE = 5.0

_C.LOGGING = CN()
_C.LOGGING.DO_TENSORBOARD_LOGGING = True
_C.LOGGING.LOGGING_ITER_INTERVAL = 10

# Low-level settings.
# These primarily include conditional optimizations that only apply to specific requirements. These likely will not be changed.
_C.BACKEND = CN()
_C.BACKEND.CUDNN_BENCHMARK = False  # Might help if input sizes are kept constant, and model does not contain conditionals.
_C.BACKEND.DATASET_DOWNLOAD_FOLDER = "./dataset"  # Only relevant if the datasets used are torchvision datasets with download support.
_C.BACKEND.DATALOADER_PIN_MEMORY = False
# As per most modern publications, the learning rate is scaled based on the batch size. absolute_lr = base_lr * total_batch_size / 256
_C.BACKEND.SCALE_LEARNING_RATE = False
_C.BACKEND.BREAK_WHEN_LOSS_NAN = True
