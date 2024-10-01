import gc
import itertools
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import custom_torchvision
from dataloader import get_cifar10_loaders, get_cifar100_loaders, get_resnet152_cifar10_tuned_weights, get_resnet152_cifar100_tuned_weights
from utils import get_device, set_env

set_env(seed=41)
output_path = Path.cwd() / "data" / "benchmark.jsonl"

batch_size = 8
gradient_accumulation_steps = 8
train_val_ratio = 0.8

cifar10_classes, cifar10_trainloader, cifar10_valloader, cifar10_testloader = get_cifar10_loaders(batch_size, train_val_ratio)
cifar100_classes, cifar100_trainloader, cifar100_valloader, cifar100_testloader = get_cifar100_loaders(batch_size, train_val_ratio)
cifar10_weights = get_resnet152_cifar10_tuned_weights()
cifar100_weights = get_resnet152_cifar100_tuned_weights()


def eval(config: dict):
    if config["dataset"] == "cifar10":
        classes, testloader, weights = cifar10_classes, cifar100_testloader, cifar10_weights
    elif config["dataset"] == "cifar100":
        classes, testloader, weights = cifar100_classes, cifar100_testloader, cifar100_weights
    device = get_device(disable_mps=False)
    model = custom_torchvision.get_custom_resnet152(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights))
    model = torch.compile(model, mode="reduce-overhead")
    model.eval()
    # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    y_true, y_pred = [], []
    with torch.inference_mode(), torch.amp.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=(torch.cuda.is_available())):
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = custom_torchvision.get_cross_max_consensus(outputs=outputs, k=2)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "config": config,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    print(results)


if __name__ == "__main__":

    def is_cached(config: dict):
        if not output_path.exists():
            return False
        content = output_path.read_text()
        lines = content.split("\n")
        for line in lines:
            if not line:
                continue
            result = json.loads(line)
            if result["config"] == config:
                return True

    searchspace = {
        "dataset": ["cifar10", "cifar100"],
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    print(f"combinations: {len(combinations)}")

    for combination in combinations:
        if is_cached(combination):
            print(f"skipping: {combination}")
            continue

        print(f"evaluating: {combination}")
        eval(config=combination)
