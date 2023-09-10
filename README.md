# pytorch-image-classification-template
A basic template for image classification via deep learning through the PyTorch framework, made to be modular and filled with common bag-of-tricks that help with performance.

# Features
- timm model integration
- experiment reproducibility via YACS config dumps
- modular codebase in PyTorch/timm/torchvision
- "bag-of-tricks" like Mixup/Cutmix implemented

# Quickstart
## Environment Setup
1. Setup a Python environment. This codebase was tested using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python v3.10, which can be set up in the following method.
```shell
# If you don't have conda installed, 
# you can setup miniconda using the following commands on Linux.
# $ mkdir -p ~/miniconda3
# $ wget https://repo.anaconda.com/miniconda/$ Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# $ rm -rf ~/miniconda3/miniconda.sh
$ conda create -n torch python=3.10
$ conda activate torch
```

2. Install PyTorch following the instructions on the PyTorch website (https://pytorch.org/get-started/locally/).
```shell
# This was the command used for testing locally, but your exact command may differ.
$ conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

3. Install the required libraries.
```shell
$ pip install -r requirements.txt
```

4A. Run training. A test script for CIFAR10 using ConvNeXt-Atto is provided.
```shell
# This command executes training on the CPU, as defined in the config file itself.
$ python train.py --cfg configs/config_quickstart.yaml
```

4B. Training can also be executed on a single GPU if a CUDA-capable CPU is available.
```shell
# Any key/value pairs defined in the YAML config can be overriden via the CLI.
# The config dump to the experiment workdir will reflect the overrides.
$ python train.py \
    --cfg configs/config_quickstart.yaml \
    --overrides EXPERIMENT.DEVICE cuda:0
```

## Training
Training primarily works with two components, the training script (`train.py`), and the YAML config file based on YACS (https://github.com/rbgirshick/yacs). YACS has the benefit of allowing for overrides via the CLI as shown in step 4B above, which allows for more convenient experimentation, particularly on job-based clusters.

A few example config files are provided in the `configs` folder, mostly consisting of different [Imagenette](https://github.com/fastai/imagenette) attempts. The following section describes how to train it.
```shell
$ python train.py --cfg configs/config_imagenette_seresnext50_5epochs.yaml
```

This simply executes the training, with the outputs being saved to the `EXPERIMENT.WORKDIR` key defined in the config file. By opening up the config file, you'll see that the target workdir is `./workdir_imagenette_seresnext50`, which is where the model logs and checkpoints get created.
```yaml
EXPERIMENT:
  DEVICE: cuda:0
  SEED: 3407
  WORKDIR: ./workdir_imagenette_seresnext50
TRAINING:
  ...
  EPOCHS: 5
```

An example for long-term training with an 80-epoch schedule is provided in `configs/config_imagenette_seresnext50_80epochs.yaml`, with settings modified to make use of the longer schedule. However, you can also use CLI overrides to extend the training defined above.

```shell
$ python train.py \
    --cfg configs/config_imagenette_seresnext50_5epochs.yaml \
    --overrides \
        TRAINING.EPOCHS 80
```

In the example above, we override the 5-epoch schedule defined in the YAML file with an 80-epoch schedule.

Expanding on 4B, you can override multiple key/value pairs at the same time, using the following format: `--overrides KEY1 value1 KEY2 value2 KEY3 value3`. Here, let's add a few regularization tricks to make use of the longer schedule, and also save things into an alternate workdir so we can compare runs against each other.

```shell
$ python train.py \
    --cfg configs/config_imagenette_seresnext50_5epochs.yaml \
    --overrides \
        EXPERIMENT.WORKDIR ./workdir_imagenette_seresnext50_long \
        TRAINING.EPOCHS 80 \       # Set training schedule to 80 epochs
        BAG_OF_TRICKS.MIXUP_CUTMIX.MIXUP_CUTMIX.DO_CUTMIX true \   # Turn on mixup and cutmix.
        BAG_OF_TRICKS.MIXUP_CUTMIX.CUTMIX_ALPHA 0.1 \
        OPTIMIZER.WEIGHT_DECAY 0.001 \      # Turn on weight decay.
        LR_SCHEDULER.TYPE CosineAnnealingWithWarmup     # Switch the step-decay into cosine annealing.
```

For more details about the different configuration key/value pairs, see the comments in `./configs/config_documentation.yaml`.

Finally, we can also add some functionality to resume experiments in case anything terminates training midway through. The training code will always save the latest checkpoint after every epoch as `{WORKDIR}/checkpoints/last.ckpt`.

```shell
$ python train.py \
    --cfg configs/config_imagenette_seresnext50_5epochs.yaml \
    --overrides \
        EXPERIMENT.WORKDIR ./workdir_imagenette_seresnext50_long \
        ...
    --ckpt ./workdir_imagenette_seresnext50_long/checkpoints/last.ckpt
```

When the `--ckpt` argument is passed, the training script will attempt to resume from the `.ckpt` file provided.

Finally, we save Tensorboard and CSV logs by default, and you can quickly compare runs. Since `tensorboard` is included in the installation, we can run the following command to compare things in Tensorboard.

```shell
$ tensorboard --logdir ./
```

## Running Model Inference
Once a model is completely trained, you can run the `infer_single_image.py` file to test your trained model on an image file.

```shell
$ python infer_single_image.py --cfg workdir/config.yaml --img path/to/image.jpg
```

This will print the class with the highest probability in the terminal, while also giving you a breakdown of the other class probabilities.

## 'Bag-of-tricks'
Techniques like [MixUp](https://arxiv.org/abs/1710.09412), [CutMix](https://arxiv.org/abs/1905.04899), [stochastic depth](https://github.com/huggingface/pytorch-image-models/blob/a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models/layers/drop.py#L140), [gradient clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html), [exponential moving averages (EMA)](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html) and [AutoAugment](https://arxiv.org/abs/1805.09501) are built into the training process, and can be toggled on and off for potential improvements.

These techniques are built into the training code, and are in the config file under the `BAG_OF_TRICKS` key. The implementation of most of these techniques are based on the [ConvNeXt-V2 repo](https://github.com/facebookresearch/ConvNeXt-V2), with some minor modifications to fit things into the modular system.

For more details about how these methods are implemented, you can find out more in `classification/builders/bag_of_tricks.py`, where the constructors are defined.

## Constructors/Builders and Custom Components
The training script primarily relies on 'builder' functions, which help define 'build' each component of the training loop. These definitions can be found under `classification/builders`.

For example, the deep learning model is defined using the following lines:
```python
model = build_model(
    model_name=cfg.MODEL.NAME,
    use_pretrained=cfg.MODEL.USE_PRETRAIN,
    num_classes=len(class_to_idx),
    dropout_rate=cfg.MODEL.DROPOUT_RATE,
    drop_path_rate=cfg.BAG_OF_TRICKS.STOCHASTIC_DEPTH.DROP_PATH_RATE,
)
```

This shows how we utilize the `cfg` generated after parsing the `.yaml` config file. All of these constructors are written such that they return standard PyTorch objects. For example, the `model` is a standard `nn.Module`, the `optimizer` is a `torch.optim.Optimizer`, and so on.

This means you can easily swap things out with your own custom components as necessary. For example, if you want to use a custom model that isn't available in timm, you can define it as per https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html#convolutional-layers, and replace the builder function with your own model definition.

You can also use the config variables during initialization instead of hardcoding them.

```python
model = CustomModel(num_classes=len(class_to_idx))
```

## Tricks and Convenience Functions
1) Most timm models can be used in the `MODEL.NAME` config property. To use a timm model, you can use `timm.list_models()` to find all the available models, and append the `timm_` prefix to the model name to use them.

2) Most `torchvision` classification datasets are accessible (with the exception of ImageNet). You can use any torchvision dataset listed in https://pytorch.org/vision/stable/datasets.html#image-classification by changing the `DATA.DATASET` config property to the dataset name with the `torchvision_` prefix. For example, to use CIFAR10, set it to `torchvision_cifar10`.

3) If you already have labelled data in an [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)-compatible dataset, set the `DATA.DATASET` config property to the filepath. If your images are already split into `train` and `val` folders, you can set `DATA.PRE_SPLIT_TRAIN_VAL` to `true`.

4) A validation set is generally recommended, and all metrics from `torcheval` can be used for model selection via the `EVALUATION.METRIC` config property. These include:

    ```
    accuracy, recall, precision, f1_score, macro_accuracy, macro_recall, macro_precision, macro_f1_score, macro_auc_roc
    ```

# Introduction
## Why make this?
The idea for this codebase was mainly for my own convenience. 

Despite image classification being one of the most iconic deep learning tasks, its not easy to find a simple codebase that lends itself to fast experimentation and quick modularization.

Sometimes, when I find some cool new technique that promises improved performance (such as [Google Brain's Lion Optimizer](https://arxiv.org/abs/2302.06675) or [FAIR's ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)), I rarely find the time to test things out on my own datasets or workflows.

A lot of this comes down to time. There's rarely a 'slot-in' solution that allows me to work with a familiar ML setup without getting into the nitty-gritty details. This is particularly true in an industry dominated by excellent frameworks like [OpenMMLab](https://github.com/open-mmlab), which abstracts away so much that its hard to make small changes without diving into the source code.

Even when I decide to write my own training scripts, it's frustrating to chain together all of the tricks-of-the-trade that make modern deep learning so powerful. While PyTorch provides [an excellent tutorial on writing a training loop](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html), it doesn't teach you how to do gradient accumulation, exponential moving averaging, automatic mixed precision (AMP), and so on.

## What exactly is this?
That's where this codebase comes in. At the base level, this is a simple script that enables simple experimentation. If you have some labelled images, you should be able to get a decent model built around [PyTorch](https://pytorch.org/) and [timm](https://github.com/huggingface/pytorch-image-models).

However, it is also possible to add your own custom components, such as new models, alternate optimizers, or custom dataloader pipelines. The code is written in vanilla PyTorch (with timm handling some of the heavy lifting), which means any logic can be easily overridden without any complex hacks or abstracted APIs.

This solves the issue of experimentation I described above. Now, instead of needing to code up a solution from scratch, I should (hopefully) be able to directly do experiments without hacking together a working codebase from scratch every single time.

# Caveats
## No Distributed Training
Unfortunately, I could not test distributed training. The training script is strictly written for single-GPU training. It might be possible to extend the script using [torch.distributed](https://pytorch.org/docs/stable/distributed.html), but that is beyond the scope of my abilities.

## Not all torchvision datasets are tested
While most of the `torchvision` classification datasets are implemented in the dataset builder, most are untested due to storage constraints. If something isn't working there, do let me know.

## Custom datasets need to have a class_to_idx property
The codebase is written in such a way where the `class_to_idx` property is a must-have for datasets, as it will be used to produce the class-confidence printouts during inference.

If you are using the native ImageDataset method to load your images, or if you are working with torchvision datasets, this shouldn't be an issue.

However, if you are working with custom code, you will need to add this property to your Dataset class.

## Imagenette performance
Experiments were conducted with the [160px variant of Imagenette](https://github.com/fastai/imagenette), resized to 128px to match leaderboard settings. These settings were not hyperparameter tuned, but obtained through guesswork.

1. `config_imagenette_seresnext50_5epochs.yaml` : Val acc of 88.457%
2. `config_imagenette_seresnext50_80epochs.yaml`: Val acc of 91.572% (Best epoch: 91.709%)
