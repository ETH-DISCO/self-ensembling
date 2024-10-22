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

from datasets import load_dataset
from torchvision.models import resnet152, ResNet152_Weights
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
from tqdm import tqdm

assert torch.cuda.is_available()

#
# config
#

args = SimpleNamespace(classes=100)

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
# preprocessing
#

def custom_rand(input_tensor, size):
    return torch.Tensor(np.random.rand(*size)).to("cuda")


def custom_choices(items, tensor):
    return np.random.choice(items, (len(tensor)))


resolutions = [32, 16, 8, 4]  # pretty arbitrary
down_noise = 0.2  # noise standard deviation to be added at the low resolution
up_noise = 0.2  # noise stadard deviation to be added at the high resolution
jit_size = 3  # max size of the x-y jit in each axis, sampled uniformly from -jit_size to +jit_size inclusive
shuffle_image_versions_randomly = False  # random shuffling of multi-res images (false in paper but good for experiments)


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

#
# eval
#

def eval_model(model, images_in, labels_in, batch_size=128):
    all_preds = []
    all_logits = []

    with torch.no_grad():
        its = int(np.ceil(float(len(images_in)) / float(batch_size))) # iterations
        pbar = tqdm(range(its), desc="evaluation", ncols=100) # progress bar
        for it in pbar:
            i1 = it * batch_size
            i2 = min([(it + 1) * batch_size, len(images_in)])

        inputs = torch.Tensor(images_in[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
        outputs = model(inputs)

        all_logits.append(outputs.detach().cpu().numpy())
        preds = torch.argmax(outputs, axis=-1) # get the index of the max logit from self ensemble
        all_preds.append(preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    return np.sum(all_preds == labels_in), all_preds.shape[0], all_logits

weights = models.ResNet152_Weights.IMAGENET1K_V1
state_dict = weights.get_state_dict(progress=True, model_dir=weights_path)


imported_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
