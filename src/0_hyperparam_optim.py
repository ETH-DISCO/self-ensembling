import itertools
import json
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import custom_torchvision
from utils import free_mem, get_device, set_seed

set_seed()


input_path = Path.cwd() / "data"
output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "dataset"


def train(config: dict):
    #
    # dataset
    #

    if config["dataset"] == "cifar10":
        classes = json.loads((config["input_path"] / "cifar10_classes.json").read_text())

        full_dataset = datasets.CIFAR10(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    elif config["dataset"] == "cifar100":
        classes = json.loads((config["input_path"] / "cifar100_classes.json").read_text())

        full_dataset = datasets.CIFAR100(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    elif config["dataset"] == "imagenet":
        classes = json.loads((config["input_path"] / "imagenet_classes.json").read_text())

    trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    #
    # load backbone
    #

    device = get_device(disable_mps=False)
    net = custom_torchvision.resnet152_ensemble(num_classes=len(classes))
    custom_torchvision.set_resnet_weights(net, models.ResNet152_Weights.IMAGENET1K_V1)
    custom_torchvision.freeze_backbone(net)
    net = net.to(device)  # don't compile: breaks on mps arch, speedup is insignificant 
    net.train()

    #
    # train loop
    #

    criterion = torch.nn.CrossEntropyLoss().to(device)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])  # see: https://karpathy.github.io/2019/04/25/recipe/
    ensemble_size = len(net.fc_layers)
    train_size = len(trainloader)

    def training_step(outputs, labels):
        losses = []
        for i in range(ensemble_size):
            loss = criterion(outputs[:, i, :], labels)
            losses.append(loss)
            running_losses[i] += loss.item()
        total_loss = sum(losses)
        total_loss.backward()
        optimizer.step()
        return losses

    for epoch in range(config["num_epochs"]):
        running_losses = [0.0] * ensemble_size
        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=train_size):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            losses = training_step(outputs=outputs, labels=labels)
            for i in range(ensemble_size):
                running_losses[i] += losses[i].item()

            if batch_idx % (train_size // 20) == 0:
                print(f"[epoch {epoch + 1}, batch {batch_idx + 1:5d}/{train_size}] ensemble loss: {', '.join(f'{l:.3f}' for l in running_losses)}")
                running_losses = [0.0] * ensemble_size

    #
    # validation loop
    #

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
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
    with open(output_path / "config.json", "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    #
    # grid search
    #

    # searchspace = {
    #     "dataset": ["cifar10", "cifar100", "imagenet"],
    #     "batch_size": [64, 128, 256, 512],
    #     "lr": [1e-4, 1e-3, 1e-2],
    #     "num_epochs": [5, 10, 15],
    #     "crossmax_k": [1, 2, 3],
    # }
    # combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    # for combination in combinations:
    #     if (output_path / "config.json").exists() and (combination in json.loads((output_path / "config.json").read_text())):
    #         print(f"skipping cached combination: {combination}")
    #         continue
    #     train(config=combination)

    config = {
        "dataset": "cifar10",
        "batch_size": 256,
        "lr": 1e-4,
        "num_epochs": 2,
        "crossmax_k": 2,
    }
    train(config=config)
