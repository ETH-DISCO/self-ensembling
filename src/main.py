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
args.add_argument("--eval", type=bool, required=False, default=False)  # True, False

args.add_argument("--dataset", type=str, required=False, default="cifar10")  # cifar10, cifar100, imagenet
# args.add_argument("--attack", type=str, required=False)  # none, fgsm, pgd

args.add_argument("--batch_size", type=int, required=False, default=256)  # 64, 128, 256, 512
args.add_argument("--lr", type=float, required=False, default=1e-4)  # 1e-4, 1e-3, 1e-2
args.add_argument("--num_epochs", type=int, required=False, default=2)  # 5, 10, 15
args.add_argument("--crossmax_k", type=int, required=False, default=2)  # 1, 2, 3
config = vars(args.parse_args())

dataset_path = Path.cwd() / "dataset"
input_path = Path.cwd() / "data"
output_path = Path.cwd() / "data"

#
# dataset
#

if config["dataset"] == "cifar10":
    classes = json.loads((input_path / "cifar10_classes.json").read_text())
    loader = datasets.CIFAR10
elif config["dataset"] == "cifar100":
    classes = json.loads((input_path / "cifar100_classes.json").read_text())
    loader = datasets.CIFAR100
elif config["dataset"] == "imagenet":
    classes = json.loads((input_path / "imagenet_classes.json").read_text())
    loader = datasets.ImageNet

full_dataset = loader(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
print(f"train size: {len(train_dataset)}")
print(f"val size: {len(val_dataset)}")

test_dataset = loader(root=dataset_path, train=False, transform=custom_torchvision.preprocess, download=True)
testloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
print(f"test size: {len(test_dataset)}")

def train():
    #
    # load backbone
    #

    device = get_device(disable_mps=False)
    net = custom_torchvision.resnet152_ensemble(num_classes=len(classes))
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
                print(f"[epoch: {epoch + 1}, batch: {batch_idx + 1:5d}/{train_size}] ensemble losses: {', '.join(f'{l:.3f}' for l in running_losses)}")
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
    with open(output_path / "config.json", "w") as f:
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
    net = custom_torchvision.resnet152_ensemble(num_classes=len(classes))
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
    with open(output_path / "eval.json", "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    if not config["eval"]:
        train()
    else:
        eval()
