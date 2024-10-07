import itertools
import json
from pathlib import Path

import torch
from autoattack import AutoAttack
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import custom_torchvision
from dataloader import get_cifar10_loaders, get_cifar100_loaders, get_resnet152_cifar10_tuned_weights, get_resnet152_cifar100_tuned_weights
from utils import get_device, set_env

set_env(seed=41)
output_path = Path.cwd() / "data" / "benchmark.jsonl"
assert torch.cuda.is_available(), "cuda is not available (infeasible to run on cpu)"

batch_size = 8

cifar10_classes, cifar10_trainloader, cifar10_valloader, cifar10_testloader = get_cifar10_loaders(batch_size, train_ratio=0.8)
cifar100_classes, cifar100_trainloader, cifar100_valloader, cifar100_testloader = get_cifar100_loaders(batch_size, train_ratio=0.8)
cifar10_weights = get_resnet152_cifar10_tuned_weights()
cifar100_weights = get_resnet152_cifar100_tuned_weights()


class AutoattackWrapper(torch.nn.Module):
    def __init__(self, model, k):
        super().__init__()
        self.model = model
        self.k = k

    def forward(self, x):
        # crossmax consensus returns [batch_size] as output
        # autoattack expects 2 dimensional [batch_size, num_classes] as output
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        outputs = self.model(x)
        preds = custom_torchvision.get_cross_max_consensus(outputs=outputs, k=self.k)
        one_hot = torch.zeros(preds.size(0), outputs.size(-1), device=preds.device)
        one_hot.scatter_(1, preds.unsqueeze(1), 1)
        return one_hot.requires_grad_(True)


def eval(config: dict):
    if config["dataset"] == "cifar10":
        classes, testloader, weights = cifar10_classes, cifar10_testloader, cifar10_weights
    elif config["dataset"] == "cifar100":
        classes, testloader, weights = cifar100_classes, cifar100_testloader, cifar100_weights

    device = get_device(disable_mps=True)
    model = custom_torchvision.get_custom_resnet152(num_classes=len(classes)).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()

    autoattack_model = AutoattackWrapper(model, k=2).to(device)
    custom_torchvision.unfreeze_backbone(autoattack_model)
    autoattack_model.eval()
    adversary = AutoAttack(autoattack_model, norm="Linf", eps=8 / 255, version="standard", device=device, verbose=True)

    y_true, y_preds, y_final = [], [], []
    for images, labels in tqdm(testloader):
        images, labels = images.to(device), labels.to(device)

        adv_images = adversary.run_standard_evaluation(images, labels, bs=batch_size)
        adv_images = adv_images.detach()
        with torch.inference_mode(), torch.amp.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=(torch.cuda.is_available())):
            predictions = model(adv_images)

        y_true.extend(labels.cpu().numpy())
        y_preds.extend(predictions.cpu().numpy())
        y_final.extend(custom_torchvision.get_cross_max_consensus(outputs=predictions, k=2).cpu().numpy())
    results = {
        **config,
        "labels": y_true,
        "predictions": y_preds,
        "final_predictions": y_final,
        "accuracy": accuracy_score(y_true, y_final),
        "precision": precision_score(y_true, y_final, average="weighted"),
        "recall": recall_score(y_true, y_final, average="weighted"),
        "f1_score": f1_score(y_true, y_final, average="weighted"),
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
