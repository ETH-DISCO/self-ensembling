import json
from itertools import product
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

    covered = 0
    for comb in tqdm(combs, total=len(combs), ncols=80):
        comb = {k: v for k, v in zip(combinations.keys(), comb)}
        if is_cached(get_current_dir() / "aggregated.jsonl", comb):
            covered += 1
    print(f"\ncovered: {covered}/{len(combs)}")
