"""
how well do we perform on the test set with adversarial attacks (compared against the baseline)?
"""

import itertools
import json
from pathlib import Path

import torch
import torchvision
from autoattack import AutoAttack
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import custom_torchvision
import dataloader
import utils

assert torch.cuda.is_available(), "cuda is not available"

utils.set_env(seed=41)
output_path = Path.cwd() / "data" / "benchmark.jsonl"

batch_size = 8

cifar10_classes, cifar10_trainloader, cifar10_valloader, cifar10_testloader = dataloader.get_cifar10_loaders(batch_size, train_ratio=0.8)
cifar100_classes, cifar100_trainloader, cifar100_valloader, cifar100_testloader = dataloader.get_cifar100_loaders(batch_size, train_ratio=0.8)
cifar10_weights = dataloader.get_resnet152_cifar10_tuned_weights()
cifar100_weights = dataloader.get_resnet152_cifar100_tuned_weights()


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


def eval(config: dict):
    #
    # data
    #

    if config["dataset"] == "cifar10":
        classes, testloader, weights = cifar10_classes, cifar10_testloader, cifar10_weights
    elif config["dataset"] == "cifar100":
        classes, testloader, weights = cifar100_classes, cifar100_testloader, cifar100_weights

    #
    # models
    #

    device = utils.get_device(disable_mps=True)

    # self-ensembling model
    model = custom_torchvision.get_custom_resnet152(num_classes=len(classes)).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    # adversary
    atk_model = AutoattackWrapper(model, k=2).to(device)
    custom_torchvision.unfreeze_backbone(atk_model)
    atk_model.eval()
    model_adversary = AutoAttack(atk_model, norm="Linf", eps=8 / 255, version="standard", device=device, verbose=True)

    # baseline model
    baseline = torchvision.models.resnet152(pretrained=False, num_classes=len(classes)).to(device)
    baseline.load_state_dict(weights, strict=True)
    baseline.eval()
    # adversary
    baseline_adversary = AutoAttack(baseline, norm="Linf", eps=8 / 255, version="standard", device=device, verbose=True)

    #
    # benchmark
    #

    y_true, y_preds_model, y_preds_baseline = [], [], []
    with torch.amp.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=(torch.cuda.is_available())):
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)

            model_adv_images = model_adversary.run_standard_evaluation(images, labels, bs=batch_size)
            model_adv_images = model_adv_images.detach()
            with torch.inference_mode():
                model_predictions = model(model_adv_images)

            baseline_adv_images = baseline_adversary.run_standard_evaluation(images, labels, bs=batch_size)
            baseline_adv_images = baseline_adv_images.detach()
            with torch.inference_mode():
                baseline_predictions = baseline(baseline_adv_images)

            y_true.extend(labels.cpu().numpy())

            y_preds_baseline.extend(baseline_predictions.cpu().numpy())
            y_preds_model.extend(custom_torchvision.get_cross_max_consensus(model_predictions, k=2).cpu().numpy())

    results = {
        **config,
        "model_accuracy": accuracy_score(y_true, y_preds_model),
        "model_precision": precision_score(y_true, y_preds_model, average="weighted"),
        "model_recall": recall_score(y_true, y_preds_model, average="weighted"),
        "model_f1_score": f1_score(y_true, y_preds_model, average="weighted"),
        "baseline_accuracy": accuracy_score(y_true, y_preds_baseline),
        "baseline_precision": precision_score(y_true, y_preds_baseline, average="weighted"),
        "baseline_recall": recall_score(y_true, y_preds_baseline, average="weighted"),
        "baseline_f1_score": f1_score(y_true, y_preds_baseline, average="weighted"),
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    searchspace = {
        "dataset": ["cifar10", "cifar100"],
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    for combination in combinations:
        print(f"evaluating: {combination}")
        eval(config=combination)
