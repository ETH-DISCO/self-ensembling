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

batch_size = 512

cifar10_classes, cifar10_trainloader, cifar10_valloader, cifar10_testloader = get_cifar10_loaders(batch_size, train_ratio=0.8)
cifar100_classes, cifar100_trainloader, cifar100_valloader, cifar100_testloader = get_cifar100_loaders(batch_size, train_ratio=0.8)
cifar10_weights = get_resnet152_cifar10_tuned_weights()
cifar100_weights = get_resnet152_cifar100_tuned_weights()


def get_cross_maxed_model(model: torch.nn.Module, k: int):
    class SingleOutputModel(torch.nn.Module):
        def __init__(self, model, k):
            super().__init__()
            self.model = model
            self.k = k

        def forward(self, x):
            outputs = self.model(x)
            consensus = custom_torchvision.get_cross_max_consensus(outputs, self.k)
            return consensus.unsqueeze(1)  # ensure 2D output

    return SingleOutputModel(model, k)


def eval(config: dict):
    if config["dataset"] == "cifar10":
        classes, testloader, weights = cifar10_classes, cifar10_testloader, cifar10_weights
    elif config["dataset"] == "cifar100":
        classes, testloader, weights = cifar100_classes, cifar100_testloader, cifar100_weights

    device = get_device(disable_mps=True)
    model = custom_torchvision.get_custom_resnet152(num_classes=len(classes)).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()

    simple_model = get_cross_maxed_model(model=model, k=2)
    simple_model.eval()
    adversary = AutoAttack(simple_model, norm="Linf", eps=8 / 255, version="standard", device=device)

    y_true, y_preds, y_final = [], [], []
    with torch.inference_mode(), torch.amp.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=(torch.cuda.is_available())):
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)

            adv_images = adversary.run_standard_evaluation(images, labels, bs=batch_size)
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
    searchspace = {"dataset": ["cifar10", "cifar100"]}
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    for combination in combinations:
        print(f"evaluating: {combination}")
        eval(config=combination)
