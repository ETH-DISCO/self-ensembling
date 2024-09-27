"""
improvement: single gpu implementation, without early stopping, lots of memory usage optimizations
"""

import itertools
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import GradScaler
from tqdm import tqdm

import custom_torchvision
from dataloader import get_cifar10_loaders, get_cifar100_loaders
from utils import free_mem, get_device, set_env

set_env(seed=41)
free_mem()
assert torch.cuda.is_available(), "cuda is not available"

#
# config constants
#

output_path = Path.cwd() / "data" / "hyperparams.jsonl"

batch_size = 16  # lower vals increase perf (128 barely fits in gpu memory)
train_val_ratio = 0.8  # common default

cifar10_classes, cifar10_trainloader, cifar10_valloader, cifar10_testloader = get_cifar10_loaders(batch_size, train_val_ratio)
cifar100_classes, cifar100_trainloader, cifar100_valloader, cifar100_testloader = get_cifar100_loaders(batch_size, train_val_ratio)


def train(config: dict):
    if config["dataset"] == "cifar10":
        classes = cifar10_classes
        trainloader = cifar10_trainloader
        valloader = cifar10_valloader
    elif config["dataset"] == "cifar100":
        classes = cifar100_classes
        trainloader = cifar100_trainloader
        valloader = cifar100_valloader

    device = get_device(disable_mps=True)
    model = custom_torchvision.get_custom_resnet152(num_classes=len(classes))
    custom_torchvision.set_imagenet_backbone(model)
    custom_torchvision.freeze_backbone(model)
    model = model.to(device)
    model = torch.compile(model, mode="reduce-overhead")

    #
    # train loop
    #

    scaler = GradScaler(device="cuda", enabled=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])  # safe bet (but adam-w is better)
    ensemble_size = len(model.fc_layers)
    train_size = len(trainloader)

    for epoch in range(config["num_epochs"]):
        # epoch train
        model.train()
        running_losses = [0.0] * ensemble_size
        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=train_size):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=(device if "cuda" in str(device) else "cpu"), enabled=("cuda" in str(device))):
                outputs = model(inputs)

            def training_step(outputs, labels):
                losses = []
                for i in range(ensemble_size):
                    loss = criterion(outputs[:, i, :], labels)
                    losses.append(loss)
                    running_losses[i] += loss.item()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                total_loss = sum(losses)
                total_loss.backward()
                optimizer.step()
                return losses

            losses = training_step(outputs=outputs, labels=labels)
            for i in range(ensemble_size):
                running_losses[i] += losses[i].item()

            if batch_idx % (train_size // 3) == 0:
                print(f"[epoch {epoch + 1}/{config['num_epochs']}: {batch_idx + 1}/{train_size}] ensemble loss: {', '.join(f'{l:.3f}' for l in running_losses)}")
                running_losses = [0.0] * ensemble_size

            free_mem()

        free_mem()

    #
    # validation loop
    #

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad(), torch.inference_mode(), torch.amp.autocast(device_type=(device if "cuda" in str(device) else "cpu"), enabled=("cuda" in str(device))):
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = custom_torchvision.get_cross_max_consensus(outputs=outputs, k=config["crossmax_k"])
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            free_mem()

    results = {
        "config": config,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    print(f"validation accuracy: {results['accuracy']:.3f}")
    with open(output_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    #
    # saving
    #

    # Saving
    # torch.save({
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scaler': scaler.state_dict(),
    # }, 'checkpoint.pth')

    # Loading
    # checkpoint = torch.load('checkpoint.pth')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scaler.load_state_dict(checkpoint['scaler'])


if __name__ == "__main__":
    #
    # grid search
    #
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
        "dataset": ["cifar10", "cifar100"],  # paper used pretrained imagenet weights with cifar10, cifar100
        "lr": [1e-1, 1e-4, 1e-7],  # paper found 1.7e-5 to be most robust
        "num_epochs": [4, 8, 16],  # paper only used 1 epoch
        "crossmax_k": [2, 3],  # 2 is the classic vickery consensus
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    print(f"searching {len(combinations)} combinations")

    for combination in combinations:
        if is_cached(combination):
            print(f"skipping: {combination}")
            continue

        print(f"training: {combination}")
        train(config=combination)
