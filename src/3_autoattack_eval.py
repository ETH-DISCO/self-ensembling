from pathlib import Path

import torch

cifar10_path = Path.cwd() / "datasets" / "restnet152_advx_individual_cifar10.pth"
data = torch.load(cifar10_path, weights_only=True)
apgd_ce = data["apgd-ce"]
apgd_t = data["apgd-t"]

print(apgd_ce)
