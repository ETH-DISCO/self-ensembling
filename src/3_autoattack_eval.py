from robustbench.data import CORRUPTIONS_DICT, get_preprocessing, load_clean_dataset, CORRUPTION_DATASET_LOADERS
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, load_model, parse_args, update_json
from robustbench.model_zoo import model_dicts as all_models


from pathlib import Path

import torch

cifar10_path = Path.cwd() / "datasets" / "restnet152_advx_individual_cifar10.pth"
data = torch.load(cifar10_path, weights_only=True)
apgd_ce = data["apgd-ce"]
apgd_t = data["apgd-t"]

print(apgd_ce)
