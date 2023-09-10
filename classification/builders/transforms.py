import torchvision
from torch import nn
import torchvision.transforms as T

from timm.data import create_transform

from typing import Tuple


def build_train_transforms(
    image_size: int,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.0,
    color_jitter_prob: float = 0.4,
    auto_augment_policy: str = None,
    interpolation_mode: str = "bilinear",
    random_erase_prob: float = 0.0,
    random_erase_mode: str = "const",
    random_erase_count: int = 1,
    pixel_mean: Tuple[float] = [0.0, 0.0, 0.0],
    pixel_std: Tuple[float] = [1.0, 1.0, 1.0],
    no_aug: bool = False,
) -> nn.Module:
    """Helper function to construct transformation chains.

    This should generally be handled manually, but this attempts to construct industry standard chains, which might include 'bag-of-tricks' methods which might not come included out of the box. Default values come from the ConvNeXt-V2 repository from FAIR (https://github.com/facebookresearch/ConvNeXt-V2/blob/main/datasets.py).

    Args:
        image_size (int): Expected input size of the network.
        hflip_prob (float, optional): Probability for horizontal flipping. Defaults to 0.5.
        vflip_prob (float, optional): Probability for vertical flipping. Defaults to 0.0.
        color_jitter_prob (float, optional): Probability for color jittering. Defaults to 0.4.
        auto_augment_policy (str, optional): Auto-augmentation policy, examples being 'rand-m9-mstd0.5-inc1'. Defaults to None.
        interpolation_mode (str, optional): Interpolation mode for image resizing. Defaults to "bilinear".
        random_erase_prob (float, optional): Probability for random erasing. Defaults to 0.0.
        random_erase_mode (str, optional): Mode for random erasing. Defaults to "const".
        random_erase_count (int, optional): Count for random erasing. Defaults to 1.
        pixel_mean (Tuple[float], optional): Pixel mean to use for image normalization. Defaults to [0.0, 0.0, 0.0].
        pixel_std (Tuple[float], optional): Pixel standard deviation to use for image normalization. Defaults to [1.0, 1.0, 1.0].
        no_aug (bool, optional): Flag to bypass all augmentations. If True, will return a blank augmentation chain. Defaults to False.

    Returns:
        nn.Module: Initialized augmentation chain.
    """
    # Generally, stick to the default timm method, which is used for ImageNet training.
    augmentation_chain = create_transform(
        input_size=image_size,
        hflip=hflip_prob,
        vflip=vflip_prob,
        is_training=True,
        color_jitter=color_jitter_prob,
        auto_augment=auto_augment_policy,
        interpolation=interpolation_mode,
        re_prob=random_erase_prob,
        re_mode=random_erase_mode,
        re_count=random_erase_count,
        mean=pixel_mean,
        std=pixel_std,
        no_aug=no_aug,
    )

    # As per the ConvNeXt-V2 repo, https://github.com/facebookresearch/ConvNeXt-V2/blob/main/datasets.py#L51
    # Add a provision that handles cases with small images, such as CIFAR10's 32-size images.
    # This is necessary as most CNNs downsample by multiple times.
    _resize_im = image_size > 32
    if not _resize_im:
        augmentation_chain.transforms[0] = T.RandomCrop(image_size, padding=4)

    return augmentation_chain


def build_eval_transforms(
    image_size: int,
    pixel_mean: Tuple[float] = [0.0, 0.0, 0.0],
    pixel_std: Tuple[float] = [1.0, 1.0, 1.0],
    crop_pct: float = None,
) -> nn.Module:
    """Evaluation transforms adapted from ConvNeXt-V2 (https://github.com/facebookresearch/ConvNeXt-V2/blob/main/datasets.py#L50C17-L50C17).

    Args:
        image_size (int): Expected input size of the network.
        pixel_mean (Tuple[float], optional): Pixel mean to use for image normalization. Defaults to [0.0, 0.0, 0.0].
        pixel_std (Tuple[float], optional): Pixel standard deviation to use for image normalization. Defaults to [1.0, 1.0, 1.0].
        crop_pct (float, optional): Cropping percentage to use for evaluation. Defaults to None.

    Returns:
        nn.Module: Initialized augmentation chain.
    """
    t = []

    # As per the ConvNeXt-V2 repo, https://github.com/facebookresearch/ConvNeXt-V2/blob/main/datasets.py#L51
    # Add a provision that handles cases with small images, such as CIFAR10's 32-size images.
    # This is necessary as most CNNs downsample by multiple times.
    _resize_im = image_size > 32
    if _resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if image_size >= 384:
            t.append(
                T.Resize(
                    (image_size, image_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
            )
            print(f"Warping {image_size} size input images...")
        else:
            if crop_pct is None:
                crop_pct = 224 / 256
            size = int(image_size / crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            )
            t.append(T.CenterCrop(image_size))

    t.append(T.ToTensor())
    t.append(T.Normalize(pixel_mean, pixel_std))
    return T.Compose(t)
