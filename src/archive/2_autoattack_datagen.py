"""
generate adversarial images using AutoAttack library

tutorial: https://github.com/fra31/auto-attack/blob/master/autoattack/examples/eval.py
modes: https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py
"""

import itertools
import os
from pathlib import Path

import custom_torchvision
import dataloader
import torch
import utils
from autoattack import AutoAttack
from dataloader import weights_path

assert torch.cuda.is_available(), "cuda is not available"

utils.set_env(seed=41)

output_path = Path.cwd() / "datasets"
batch_size = 8
individual = True

cifar10_classes, _, _, cifar10_testloader = dataloader.get_cifar10_loaders(batch_size, train_ratio=0.8)
cifar100_classes, _, _, cifar100_testloader = dataloader.get_cifar100_loaders(batch_size, train_ratio=0.8)
cifar10_weights = dataloader.get_resnet152_cifar10_tuned_weights()
cifar100_weights = dataloader.get_resnet152_cifar100_tuned_weights()
baseline_weights = dataloader.get_resnet152_imagenet_weights()


def wrap(model: torch.nn.Module, device: torch.device):
    class AutoattackWrapper(torch.nn.Module):
        def __init__(self, model, k):
            super().__init__()
            self.model = model
            self.k = k

        def forward(self, x):
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            outputs = self.model(x)
            preds = custom_torchvision.get_cross_max_consensus_logits(outputs, k=self.k)
            return preds

    atk_model = AutoattackWrapper(model, k=2).to(device)
    custom_torchvision.unfreeze_backbone(atk_model)
    atk_model.eval()
    return atk_model


def eval(config: dict):
    if config["dataset"] == "cifar10":
        classes, testloader, weights = cifar10_classes, cifar10_testloader, cifar10_weights
    elif config["dataset"] == "cifar100":
        classes, testloader, weights = cifar100_classes, cifar100_testloader, cifar100_weights

    device = utils.get_device(disable_mps=True)

    model = custom_torchvision.get_custom_resnet152(num_classes=len(classes)).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()

    adversary = AutoAttack(wrap(model=model, device=device), norm="Linf", eps=8 / 255, version="standard", device=device, verbose=True)
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]  # paper only runs these two attacks

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    l = [x for (x, y) in testloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in testloader]
    y_test = torch.cat(l, 0)

    with torch.no_grad():
        datasetname = config["dataset"]
        if individual:
            adv_complete = adversary.run_standard_evaluation_individual(x_test, y_test, bs=batch_size)
            torch.save(adv_complete, output_path / f"restnet152_advx_individual_{datasetname}.pth")
        else:
            adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size, state_path=weights_path / "restnet152_advx_state.pth")
            torch.save({"adv_complete": adv_complete}, output_path / f"restnet152_advx_{datasetname}.pth")


if __name__ == "__main__":
    searchspace = {
        "dataset": ["cifar10", "cifar100"],
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    for combination in combinations:
        print(f"evaluating: {combination}")
        eval(config=combination)
