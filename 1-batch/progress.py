import sys
import hashlib
import json
import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm
from utils import *


def is_cached(filepath, combination):
    if filepath.exists() and filepath.stat().st_size > 0:
        lines = filepath.read_text().strip().split("\n")
        lines = [json.loads(line) for line in lines]
        for line in lines:
            if all(line[k] == v for k, v in combination.items()):
                return True
    return False


if __name__ == "__main__":
    BATCH_ID = int(sys.argv[1])
    TOTAL_BATCHES = int(sys.argv[2])
    DATASET = sys.argv[3] # cifar10, cifar100, imagenette
    assert 0 <= BATCH_ID < TOTAL_BATCHES

    combinations = {
        "dataset": ["cifar10"],
        # train config
        "train_epochs": [0, 2, 6],
        "train_hcaptcha_ratio": [0.0, 0.5, 1.0],
        "train_opacity": [0, 2, 4, 8, 16, 32, 64, 128],
        # mask config
        "mask_sides": [3, 4, 6, 10],
        "mask_per_rowcol": [2, 4, 10],
        "mask_num_concentric": [2, 5, 10],
        "mask_colors": [True, False],
    }
    combs = list(product(*combinations.values()))
    print(f"total combinations: {len(combs)}")

    # select batch
    batch_size = len(combs) // TOTAL_BATCHES
    combs = [combs[i:i+batch_size] for i in range(0, len(combs), batch_size)]
    combs = combs[BATCH_ID]

    for idx, comb in enumerate(combs):
        print(f"progress: {idx+1}/{len(combs)}")
        comb = {k: v for k, v in zip(combinations.keys(), comb)}
        if is_cached(get_current_dir() / "aggregated.jsonl", comb):
            continue
