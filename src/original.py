import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import random
import hashlib
import time
import copy

import matplotlib.pyplot as plt

from contextlib import contextmanager
from tqdm import tqdm
import os
import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets

assert torch.cuda.is_available()

#
# config
#

args = SimpleNamespace(
    classes=100
)

classes_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(classes_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)

#
# seed
#

seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


@contextmanager
def isolated_environment():
    # save and restore random states in a context manager
    # used to separate random-seed-fixing behavior from the attacks later
    np_random_state = np.random.get_state()
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_print_options = np.get_printoptions()
    try:
        yield  # execute code block
    finally:
        np.random.set_state(np_random_state)
        random.setstate(python_random_state)
        torch.set_rng_state(torch_random_state)
        if cuda_random_state:
            torch.cuda.set_rng_state_all(cuda_random_state)
        np.set_printoptions(**numpy_print_options)

#
# data
#

if args.classes == 10:
    trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    testset = datasets.CIFAR10(root=dataset_path, train=False, download=True)
    original_images_train_np = np.array(trainset.data)
    original_labels_train_np = np.array(trainset.targets)
    original_images_test_np = np.array(testset.data)
    original_labels_test_np = np.array(testset.targets)
elif args.classes == 100:
    trainset = datasets.CIFAR100(root=dataset_path, train=True, download=True)
    testset = datasets.CIFAR100(root=dataset_path, train=False, download=True)
    original_images_train_np = np.array(trainset.data)
    original_labels_train_np = np.array(trainset.targets)
    original_images_test_np = np.array(testset.data)
    original_labels_test_np = np.array(testset.targets)
else:
    raise ValueError

images_train_np = original_images_train_np / 255.0  # scale to [0, 1]
images_test_np = original_images_test_np / 255.0
labels_train_np = original_labels_train_np
labels_test_np = original_labels_test_np

#
# preprocessing & image augmentation
#


def custom_rand(input_tensor, size):
    return torch.Tensor(np.random.rand(*size)).to("cuda")


def custom_choices(items, tensor):
    return np.random.choice(items, (len(tensor)))


resolutions = [32, 16, 8, 4]  # pretty arbitrary
down_noise = 0.2  # noise standard deviation to be added at the low resolution
up_noise = 0.2  # noise stadard deviation to be added at the high resolution
jit_size = 3  # max size of the x-y jit in each axis, sampled uniformly from -jit_size to +jit_size inclusive

# to shuffle randomly which image is which in the multi-res stack
# False for all experiments in the paper, good for ablations
shuffle_image_versions_randomly = False


def default_make_multichannel_input(images):
    return torch.concatenate([images] * len(resolutions), axis=1)


def make_multichannel_input(images):
    all_channels = []

    for i, r in enumerate(resolutions):
        down_res = r
        jits_x = custom_choices(range(-jit_size, jit_size + 1), images + i)  # x-shift
        jits_y = custom_choices(range(-jit_size, jit_size + 1), 51 * images + 7 * i + 125 * r)  # y-shift
        contrasts = custom_choices(np.linspace(0.7, 1.5, 100), 7 + 3 * images + 9 * i + 5 * r)  # change in contrast
        color_amounts = contrasts = custom_choices(np.linspace(0.5, 1.0, 100), 5 + 7 * images + 8 * i + 2 * r)  # change in color amount

        def apply_transformations(images, down_res=224, up_res=224, jit_x=0, jit_y=0, down_noise=0.0, up_noise=0.0, contrast=1.0, color_amount=1.0):
            images_collected = []
            for i in range(images.shape[0]):
                image = images[i]
                image = torchvision.transforms.functional.adjust_contrast(image, contrast[i])  # contrast
                image = torch.roll(image, shifts=(jit_x[i], jit_y[i]), dims=(-2, -1))  # x, y jitter
                image = color_amount[i] * image + torch.mean(image, axis=0, keepdims=True) * (1 - color_amount[i])  # grayscaling
                images_collected.append(image)
            images = torch.stack(images_collected, axis=0)

            images = F.interpolate(images, size=(down_res, down_res), mode="bicubic")  # downscale
            noise = down_noise * custom_rand(images + 312, (images.shape[0], 3, down_res, down_res)).to("cuda")  # low res noise
            images = images + noise
            images = F.interpolate(images, size=(up_res, up_res), mode="bicubic")  # upscale
            noise = up_noise * custom_rand(images + 812, (images.shape[0], 3, up_res, up_res)).to("cuda")  # high res noise
            images = images + noise
            images = torch.clip(images, 0, 1)  # clip to right range
            return images

        images_now = apply_transformations(images, down_res=down_res, up_res=32, jit_x=jits_x, jit_y=jits_y, down_noise=down_noise, up_noise=up_noise, contrast=contrasts, color_amount=color_amounts)
        all_channels.append(images_now)

    if not shuffle_image_versions_randomly:
        return torch.concatenate(all_channels, axis=1)
    elif shuffle_image_versions_randomly:
        indices = torch.randperm(len(all_channels))
        shuffled_tensor_list = [all_channels[i] for i in indices]
        return torch.concatenate(shuffled_tensor_list, axis=1)

sample_images = images_test_np[:5]

for j in [0, 1]:
    multichannel_images = (
        make_multichannel_input(
            torch.Tensor(sample_images.transpose([0, 3, 1, 2])).to("cuda")
        )
        .detach()
        .cpu()
        .numpy()
        .transpose([0, 2, 3, 1])
    )

    N = 1 + multichannel_images.shape[3] // 3

    plt.figure(figsize=(N * 5.5, 5))

    plt.subplot(1, N, 1)
    plt.title("original")
    plt.imshow(sample_images[j])
    plt.xticks([], [])
    plt.yticks([], [])

    for i in range(N - 1):
        plt.subplot(1, N, i + 2)
        plt.title(f"res={resolutions[i]}")
        plt.imshow(multichannel_images[j, :, :, 3 * i : 3 * (i + 1)])
        plt.xticks([], [])
        plt.yticks([], [])

    plt.show()