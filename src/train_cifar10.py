"""
next steps:

- hyperparam optimization
- other datasets: cifar-100, imagenet
- threats: adversarial attacks, noisy labels, domain shift / Linf, L2, corruptions
- robustbench benchmarking + masks, visualizing results
    - https://github.com/RobustBench/robustbench#model-zoo
    - https://github.com/RobustBench/robustbench#notebooks


"""

import argparse
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


#
# config
#

args = argparse.ArgumentParser()
args.add_argument("--batch_size", type=int, required=True)  # 64, 128, 256, 512
args.add_argument("--lr", type=float, required=True)  # 1e-4, 1e-3, 1e-2
args.add_argument("--num_epochs", type=int, required=True)  # 5, 10, 15
args.add_argument("--crossmax_k", type=int, required=True)  # 1, 2, 3
hyperparams = vars(args.parse_args())

dataset_path = Path.cwd() / "dataset"
output_path = Path.cwd() / "data"

#
# dataset
#

cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

cifar10_full = datasets.CIFAR10(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
train_size = int(0.8 * len(cifar10_full))
val_size = len(cifar10_full) - train_size
cifar10_train, cifar10_val = random_split(cifar10_full, [train_size, val_size])
trainloader = DataLoader(cifar10_train, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
valloader = DataLoader(cifar10_val, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
print(f"train size: {len(cifar10_train)}")
print(f"val size: {len(cifar10_val)}")

cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, transform=custom_torchvision.preprocess, download=True)
testloader = DataLoader(cifar10_test, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
print(f"test size: {len(cifar10_test)}")


def train():
    #
    # load backbone
    #

    device = get_device(disable_mps=False)
    net = custom_torchvision.resnet152_ensemble(num_classes=len(cifar10_classes))
    custom_torchvision.set_resnet_weights(net, models.ResNet152_Weights.IMAGENET1K_V1)
    custom_torchvision.freeze_backbone(net)
    net = net.to(device)
    net.train()

    #
    # train loop
    #

    criterion = torch.nn.CrossEntropyLoss().to(device)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams["lr"])  # see: https://karpathy.github.io/2019/04/25/recipe/
    ensemble_size = len(net.fc_layers)

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

    for epoch in range(hyperparams["num_epochs"]):
        running_losses = [0.0] * ensemble_size
        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            losses = training_step(outputs=outputs, labels=labels)
            for i in range(ensemble_size):
                running_losses[i] += losses[i].item()

            if batch_idx % 20 == 19:
                print(f"[epoch: {epoch + 1}, batch: {batch_idx + 1:5d}/{len(trainloader)}] ensemble losses: {', '.join(f'{l:.3f}' for l in running_losses)}")
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
            predictions = custom_torchvision.get_cross_max_consensus(outputs=outputs, k=hyperparams["crossmax_k"])
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            free_mem()

    results = {
        "hyperparams": hyperparams,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    with open(output_path / "hyperparams.json", "w") as f:
        f.write(json.dumps(results, indent=4))

    #
    # save model
    #

    modelpath = output_path / "model.pth"
    if modelpath.exists():
        modelpath.unlink()
    torch.save(net.state_dict(), output_path / "model.pth")


def eval():
    #
    # load model
    #

    device = get_device(disable_mps=False)
    net = custom_torchvision.resnet152_ensemble(num_classes=len(cifar10_classes))
    net.load_state_dict(torch.load(output_path / "model.pth", map_location=torch.device("cpu"), weights_only=True))
    net = net.to(device)
    net.eval()

    #
    # eval loop
    #

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predictions = custom_torchvision.get_cross_max_consensus(outputs=outputs, k=hyperparams["crossmax_k"])
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            free_mem()

    results = {
        "hyperparams": hyperparams,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    with open(output_path / "eval.json", "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    train()
