import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

import custom_torchvision
from datasets import load_dataset
from utils import set_env

set_env(seed=41)

classes_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(classes_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


"""
datasets
"""


def get_cifar10_loaders(batch_size: int, train_ratio: int):
    classes_cifar10 = json.loads((classes_path / "cifar10_classes.json").read_text())

    full_dataset_cifar10 = datasets.CIFAR10(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
    train_size = int(train_ratio * len(full_dataset_cifar10))
    val_size = len(full_dataset_cifar10) - train_size
    train_dataset_cifar10, val_dataset_cifar10 = random_split(full_dataset_cifar10, [train_size, val_size])
    trainloader_cifar10 = DataLoader(train_dataset_cifar10, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    valloader_cifar10 = DataLoader(val_dataset_cifar10, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    test_dataset_cifar10 = datasets.CIFAR10(root=dataset_path, train=False, transform=custom_torchvision.preprocess, download=True)
    testloader_cifar10 = DataLoader(test_dataset_cifar10, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    return classes_cifar10, trainloader_cifar10, valloader_cifar10, testloader_cifar10


def get_cifar100_loaders(batch_size: int, train_ratio: int):
    classes_cifar100 = json.loads((classes_path / "cifar100_classes.json").read_text())

    full_dataset_cifar100 = datasets.CIFAR100(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
    train_size = int(train_ratio * len(full_dataset_cifar100))
    val_size = len(full_dataset_cifar100) - train_size
    train_dataset_cifar100, val_dataset_cifar100 = random_split(full_dataset_cifar100, [train_size, val_size])
    trainloader_cifar100 = DataLoader(train_dataset_cifar100, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    valloader_cifar100 = DataLoader(val_dataset_cifar100, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    test_dataset_cifar100 = datasets.CIFAR100(root=dataset_path, train=False, transform=custom_torchvision.preprocess, download=True)
    testloader_cifar100 = DataLoader(test_dataset_cifar100, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    return classes_cifar100, trainloader_cifar100, valloader_cifar100, testloader_cifar100


def get_imagenet_loaders(batch_size: int, train_ratio: int):
    # really slow: data loading takes ~1h, preprocessing takes ~1h, use only for final evaluation
    classes_imagenet = json.loads((classes_path / "imagenet_classes.json").read_text())

    full_dataset_imagenet = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="train", streaming=False, cache_dir=dataset_path)
    full_dataset_imagenet = [(custom_torchvision.preprocess(x["image"].convert("RGB")), x["label"]) for x in tqdm(full_dataset_imagenet)]
    train_size = int(train_ratio * len(full_dataset_imagenet))
    val_size = len(full_dataset_imagenet) - train_size
    train_dataset_imagenet, val_dataset_imagenet = random_split(full_dataset_imagenet, [train_size, val_size])
    trainloader_imagenet = DataLoader(train_dataset_imagenet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    valloader_imagenet = DataLoader(val_dataset_imagenet, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    test_dataset_imagenet = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=False, cache_dir=dataset_path)
    test_dataset_imagenet = [(custom_torchvision.preprocess(x["image"].convert("RGB")), x["label"]) for x in tqdm(full_dataset_imagenet)]
    testloader_imagenet = DataLoader(test_dataset_imagenet, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    return classes_imagenet, trainloader_imagenet, valloader_imagenet, testloader_imagenet


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_ratio):
        super().__init__()
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def setup(self, stage=None):
        self.classes, self.train_loader, self.val_loader, self.test_loader = get_cifar10_loaders(self.batch_size, self.train_ratio)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_ratio):
        super().__init__()
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def setup(self, stage=None):
        self.classes, self.train_loader, self.val_loader, self.test_loader = get_cifar100_loaders(self.batch_size, self.train_ratio)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


"""
weights
"""


def get_resnet152_imagenet_weights():
    weights = models.ResNet152_Weights.IMAGENET1K_V1
    state_dict = weights.get_state_dict(progress=True, model_dir=weights_path)
    return state_dict


def get_resnet152_cifar10_tuned_weights():
    repo_id = "sueszli/self-ensembling-resnet152"
    file_path = hf_hub_download(repo_id=repo_id, filename="model_cifar10_16epochs.safetensors", local_dir=weights_path, local_dir_use_symlinks=False)
    weights = load_file(file_path)
    return weights


def get_resnet152_cifar100_tuned_weights():
    repo_id = "sueszli/self-ensembling-resnet152"
    file_path = hf_hub_download(repo_id=repo_id, filename="model_cifar100_16epochs.safetensors", local_dir=weights_path, local_dir_use_symlinks=False)
    weights = load_file(file_path)
    return weights


get_resnet152_cifar100_tuned_weights()
