"""This contains the builder function for datasets. 

Generally, the default ImageFolder will be used, but for cases where torchvision datasets are requrested, the builder function will generate the train/test splits using the native torchvision functions.
"""

import os

import torch
from torch import nn
import torchvision

# Typing-specific imports.
from typing import Tuple, List, Union
from torchvision.datasets.vision import VisionDataset


class SubsetDataset(torch.utils.data.Dataset):
    """This class is needed for ImageFolders. Since performing a random_split on an ImageFolder dataset inherits the exact same transforms, this means we cannot have unique augmentation chains for train and eval splits. This function allows us to use different subsets.

    Reference: https://stackoverflow.com/questions/75010445/pytorch-apply-different-transform-to-dataset-after-creation
    """

    def __init__(
        self,
        subset: torch.utils.data.dataset.Subset,
        class_to_idx: dict,
        transform=None,
    ):
        self.subset = subset
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def torchvision_dataset_builder(
    dataset_name: str,
    transforms_train: nn.Module,
    transforms_eval: nn.Module,
    download_folder: str,
    split_train: float,
    split_eval: float,
) -> Tuple[VisionDataset, VisionDataset]:
    """Helper function that groups the torchvision classification datasets.

    Args:
        dataset_name (str): Name identifier of the torchvision dataset.
        transforms_train (nn.Module): Transformations to use for the training set.
        transforms_eval (nn.Module): Transformations to use for the evaluation set.
        download_folder (str): Download folder to use for dataset downloads, or the root folder if possible.
        split_train (float): Split percentage for train split. Will only be used if the dataset does not have an inherent split.
        split_eval (float): Split percentage for evaluation split. Will only be used if the dataset does not have an inherent split.

    Returns:
        Tuple[VisionDataset, VisionDataset]: The vision dataset. Will generally be in the format of (dataset_train, dataset_eval).
    """
    if dataset_name == "caltech101":
        # For caltech, splits need to be manually made, so no transforms need to be used.
        dataset = torchvision.datasets.Caltech101(
            download_folder,
            train=True,
            transform=None,
            target_type="category",
            download=True,
        )
        class_to_idx = dataset.class_to_idx

        # This dataset does not have an inherent train/test split, so we create it ourselves.
        # For local image folders, we need to perform the train/eval splits.
        dataset_train, dataset_eval = torch.utils.data.random_split(
            dataset, [split_train, split_eval]
        )

        dataset_train = SubsetDataset(dataset_train, class_to_idx, transforms_train)
        dataset_eval = SubsetDataset(dataset_eval, class_to_idx, transforms_eval)

        return dataset_train, dataset_eval
    elif dataset_name == "caltech256":
        # For caltech, splits need to be manually made, so no transforms need to be used.
        dataset = torchvision.datasets.Caltech256(
            download_folder,
            train=True,
            transform=None,
            target_type="category",
            download=True,
        )
        class_to_idx = dataset.class_to_idx

        # This dataset does not have an inherent train/test split, so we create it ourselves.
        # For local image folders, we need to perform the train/eval splits.
        dataset_train, dataset_eval = torch.utils.data.random_split(
            dataset, [split_train, split_eval]
        )

        dataset_train = SubsetDataset(dataset_train, class_to_idx, transforms_train)
        dataset_eval = SubsetDataset(dataset_eval, class_to_idx, transforms_eval)

        return dataset_train, dataset_eval
    elif dataset_name == "cifar10":
        dataset_train = torchvision.datasets.CIFAR10(
            download_folder, train=True, transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.CIFAR10(
            download_folder, train=False, transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "cifar100":
        dataset_train = torchvision.datasets.CIFAR100(
            download_folder, train=True, transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.CIFAR100(
            download_folder, train=False, transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "country211":
        dataset_train = torchvision.datasets.Country211(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.CIFAR100(
            download_folder, split="valid", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "dtd":
        dataset_train = torchvision.datasets.Country211(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.CIFAR100(
            download_folder, split="val", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "emnist":
        dataset_train = torchvision.datasets.EMNIST(
            download_folder,
            split="byclass",
            train=True,
            transform=transforms_train,
            download=True,
        )
        dataset_eval = torchvision.datasets.EMNIST(
            download_folder,
            split="byclass",
            train=False,
            transform=transforms_eval,
            download=True,
        )
        return dataset_train, dataset_eval
    elif dataset_name == "eurosat":
        # For caltech, splits need to be manually made, so no transforms need to be used.
        dataset = torchvision.datasets.EuroSAT(
            download_folder, transform=None, download=True
        )
        class_to_idx = dataset.class_to_idx

        # This dataset does not have an inherent train/test split, so we create it ourselves.
        # For local image folders, we need to perform the train/eval splits.
        dataset_train, dataset_eval = torch.utils.data.random_split(
            dataset, [split_train, split_eval]
        )

        dataset_train = SubsetDataset(dataset_train, class_to_idx, transforms_train)
        dataset_eval = SubsetDataset(dataset_eval, class_to_idx, transforms_eval)

        return dataset_train, dataset_eval
    elif dataset_name == "fashionmnist":
        dataset_train = torchvision.datasets.FashionMNIST(
            download_folder, train=True, transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.FashionMNIST(
            download_folder, train=False, transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "fer2013":
        dataset_train = torchvision.datasets.FER2013(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.FER2013(
            download_folder, split="test", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "fgvcaircraft":
        dataset_train = torchvision.datasets.FGVCAircraft(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.FGVCAircraft(
            download_folder, split="val", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "flowers102":
        dataset_train = torchvision.datasets.Flowers102(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.Flowers102(
            download_folder, split="val", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "food101":
        dataset_train = torchvision.datasets.Food101(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.Food101(
            download_folder, split="test", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "gtsrb":
        dataset_train = torchvision.datasets.GTSRB(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.GTSRB(
            download_folder, split="test", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "inaturalist":
        dataset_train = torchvision.datasets.INaturalist(
            download_folder,
            version="2021_train",
            transform=transforms_train,
            download=True,
        )
        dataset_eval = torchvision.datasets.INaturalist(
            download_folder,
            version="2021_valid",
            transform=transforms_eval,
            download=True,
        )
        return dataset_train, dataset_eval
    elif dataset_name == "kmnist":
        dataset_train = torchvision.datasets.KMNIST(
            download_folder, train=True, transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.KMNIST(
            download_folder, train=False, transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "lfwpeople":
        dataset_train = torchvision.datasets.LFWPeople(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.LFWPeople(
            download_folder, split="test", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "lsun":
        dataset_train = torchvision.datasets.LSUN(
            download_folder, classes="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.LSUN(
            download_folder, classes="val", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "mnist":
        dataset_train = torchvision.datasets.MNIST(
            download_folder, train=True, transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.MNIST(
            download_folder, train=False, transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "oxfordiiitpet":
        # For caltech, splits need to be manually made, so no transforms need to be used.
        dataset = torchvision.datasets.OxfordIIITPet(
            download_folder,
            split="trainval",
            target_types="category",
            transforms=None,
            download=True,
        )
        class_to_idx = dataset.class_to_idx

        # This dataset does not have an inherent train/test split, so we create it ourselves.
        # For local image folders, we need to perform the train/eval splits.
        dataset_train, dataset_eval = torch.utils.data.random_split(
            dataset, [split_train, split_eval]
        )

        dataset_train = SubsetDataset(dataset_train, class_to_idx, transforms_train)
        dataset_eval = SubsetDataset(dataset_eval, class_to_idx, transforms_eval)

        return dataset_train, dataset_eval
    elif dataset_name == "places365":
        dataset_train = torchvision.datasets.Places365(
            download_folder,
            split="train-standard",
            transform=transforms_train,
            download=True,
        )
        dataset_eval = torchvision.datasets.Places365(
            download_folder, split="val", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "pcam":
        dataset_train = torchvision.datasets.PCAM(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.PCAM(
            download_folder, split="val", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "qmnist":
        dataset_train = torchvision.datasets.QMNIST(
            download_folder, train=True, transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.QMNIST(
            download_folder, train=False, transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "renderedsst2":
        dataset_train = torchvision.datasets.RenderedSST2(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.RenderedSST2(
            download_folder, split="val", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "semeion":
        # For caltech, splits need to be manually made, so no transforms need to be used.
        dataset = torchvision.datasets.SEMEION(
            download_folder, transforms=None, download=True
        )
        class_to_idx = dataset.class_to_idx

        # This dataset does not have an inherent train/test split, so we create it ourselves.
        # For local image folders, we need to perform the train/eval splits.
        dataset_train, dataset_eval = torch.utils.data.random_split(
            dataset, [split_train, split_eval]
        )

        dataset_train = SubsetDataset(dataset_train, class_to_idx, transforms_train)
        dataset_eval = SubsetDataset(dataset_eval, class_to_idx, transforms_eval)

        return dataset_train, dataset_eval
    elif dataset_name == "sbu":
        # For caltech, splits need to be manually made, so no transforms need to be used.
        dataset = torchvision.datasets.SBU(
            download_folder, transforms=None, download=True
        )
        class_to_idx = dataset.class_to_idx

        # This dataset does not have an inherent train/test split, so we create it ourselves.
        # For local image folders, we need to perform the train/eval splits.
        dataset_train, dataset_eval = torch.utils.data.random_split(
            dataset, [split_train, split_eval]
        )

        dataset_train = SubsetDataset(dataset_train, class_to_idx, transforms_train)
        dataset_eval = SubsetDataset(dataset_eval, class_to_idx, transforms_eval)

        return dataset_train, dataset_eval
    elif dataset_name == "stanfordcars":
        dataset_train = torchvision.datasets.StanfordCars(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.StanfordCars(
            download_folder, split="test", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "stl10":
        dataset_train = torchvision.datasets.STL10(
            download_folder, split="train", transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.STL10(
            download_folder, split="test", transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval
    elif dataset_name == "sun397":
        # For caltech, splits need to be manually made, so no transforms need to be used.
        dataset = torchvision.datasets.SUN397(
            download_folder, transforms=None, download=True
        )
        class_to_idx = dataset.class_to_idx

        # This dataset does not have an inherent train/test split, so we create it ourselves.
        # For local image folders, we need to perform the train/eval splits.
        dataset_train, dataset_eval = torch.utils.data.random_split(
            dataset, [split_train, split_eval]
        )

        dataset_train = SubsetDataset(dataset_train, class_to_idx, transforms_train)
        dataset_eval = SubsetDataset(dataset_eval, class_to_idx, transforms_eval)

        return dataset_train, dataset_eval
    elif dataset_name == "usps":
        dataset_train = torchvision.datasets.USPS(
            download_folder, train=True, transform=transforms_train, download=True
        )
        dataset_eval = torchvision.datasets.USPS(
            download_folder, train=False, transform=transforms_eval, download=True
        )
        return dataset_train, dataset_eval


def build_dataset(
    dataset_identifier: str,
    pre_split_train_val: bool,
    split_train: float,
    split_eval: float,
    transforms_train: nn.Module,
    transforms_eval: nn.Module,
    download_folder: str,
) -> Tuple[torch.utils.data.Dataset, Union[torch.utils.data.Dataset, None]]:
    """Helper function to create the train and eval datasets.

    In the case where no eval set is available, the function will return a NoneType object as the 2nd output.

    Args:
        dataset_identifier (str): Identifier for the dataset. Can be a torchvision dataset or a local folder of images as per the torchvision ImageDataset convention.
        pre_split_train_val (bool): Whether a train/val split already exists. If True, will load the train and val sets separately. Assumes that the folder specified by dataset_identified already has a train and val folder.
        split_train (float): Percentage split for the training set.
        split_eval (float): Percentage split for the eval set. Can be set to 0.0 to return only the training set.
        transforms_train (nn.Module): Transformations for the training set.
        transforms_eval (nn.Module): Transformations for the eval set.
        download_folder (str): Download folder to use for torchvision dataset downloads.

    Returns:
        Tuple[torch.utils.data.Dataset, Union[torch.utils.data.Dataset, None]]: Returns 2 objects (dataset_train, dataset_eval), the train and eval sets. If split_eval is set to 0.0, this function will return (dataset_train, None)
    """
    # To mark torchvision classification datasets, we prepend 'torchvision_' to the dataset name.
    # Thus, if we wanted the CIFAR10 dataset, we would provide dataset_name=torchvision_cifar10
    if "torchvision_" in dataset_identifier.lower():
        # Split out the 'torchvision_' prefix from the identifier string.
        dataset_identifier = dataset_identifier.lower().split("torchvision_")[-1]
        print(f"Requesting torchvision dataset: {dataset_identifier}.")

        dataset_train, dataset_eval = torchvision_dataset_builder(
            dataset_identifier,
            transforms_train,
            transforms_eval,
            download_folder,
            split_train,
            split_eval,
        )

        # If there is no eval split, only return the training dataloader.
        if split_eval == 0.0:
            return dataset_train, None
        else:
            return dataset_train, dataset_eval
    else:
        print(f"Using local dataset as ImageFolder: {dataset_identifier}.")

        # If cfg.DATA.PRE_SPLIT_TRAIN_VAL is False, assume that the folder consists of folders corresponding to each class
        if not pre_split_train_val:
            # If there is no eval split, only return the training dataloader.
            if split_eval == 0.0:
                return (
                    torchvision.datasets.ImageFolder(
                        dataset_identifier, transform=transforms_train
                    ),
                    None,
                )
            else:
                # Initialize the main dataset without the transforms.
                # We will use the SubsetDataset class to attach the train- and eval-specific augmentations, after the splits.
                dataset = torchvision.datasets.ImageFolder(
                    dataset_identifier, transform=None
                )
                class_to_idx = dataset.class_to_idx

                # For local image folders, we need to perform the train/eval splits.
                dataset_train, dataset_eval = torch.utils.data.random_split(
                    dataset, [split_train, split_eval]
                )

                dataset_train = SubsetDataset(
                    dataset_train, class_to_idx, transforms_train
                )
                dataset_eval = SubsetDataset(
                    dataset_eval, class_to_idx, transforms_eval
                )

                return dataset_train, dataset_eval
        else:
            # This happens when we already have a pre-split train/val.
            path_train = os.path.join(dataset_identifier, "train")
            path_val = os.path.join(dataset_identifier, "val")

            if not os.path.exists(path_train):
                raise ValueError(
                    f"cfg.DATA.PRE_SPLIT_TRAIN_VAL is True, but train folder {path_train} does not exist."
                )
            if not os.path.exists(path_val):
                raise ValueError(
                    f"cfg.DATA.PRE_SPLIT_TRAIN_VAL is True, but eval folder {path_val} does not exist."
                )

            dataset_train = torchvision.datasets.ImageFolder(
                path_train, transform=transforms_train
            )

            dataset_eval = torchvision.datasets.ImageFolder(
                path_val, transform=transforms_eval
            )

            print(
                f"Using pre-split train ({len(dataset_train)}) and val ({len(dataset_eval)})."
            )
            return dataset_train, dataset_eval
