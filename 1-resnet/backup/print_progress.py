import json
import os
from itertools import product

from tqdm import tqdm
from utils import *

data_path = get_current_dir().parent / "data"
dataset_path = get_current_dir().parent / "datasets"
weights_path = get_current_dir().parent / "weights"
output_path = get_current_dir()
mask_path = get_current_dir() / "masks"

os.makedirs(data_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)

def is_cached(filepath, combination):
    if filepath.exists() and filepath.stat().st_size > 0:
        lines = filepath.read_text().strip().split("\n")
        lines = [json.loads(line) for line in lines]
        for line in lines:
            if all(line[k] == v for k, v in combination.items()):
                return True
    return False


if __name__ == "__main__":
    fpath = output_path / "resnet.jsonl"

    combinations = {
        # "dataset": ["cifar10", "cifar100", "imagenette"],
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

    for idx, comb in tqdm(enumerate(combs), total=len(combs), desc="progress", ncols=100):
        comb = {k: v for k, v in zip(combinations.keys(), comb)}
        if is_cached(fpath, comb):
            continue
        
        last_idx = idx - 1
        total = len(combs)
        print(f"\nProgress: {last_idx}/{total}")
        missing = total - last_idx
        print(f"Missing: {missing}")
        exit(0)
