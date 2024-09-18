import json
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm

import custom_torchvision
from utils import free_mem, get_device, set_seed

set_seed()

#
# hyperparams
#

hyperparams = {
    "batch_size": 128,
    "lr": 0.1,
    "num_epochs": 200,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "lr_scheduler": "cosine",
}

dataset_path = Path.cwd() / "dataset"
output_path = Path.cwd() / "data"

#
# data
#

cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
cifar10_full = datasets.CIFAR10(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
train_size = int(0.8 * len(cifar10_full))  # holdout 80-20 split

# train, val set
val_size = len(cifar10_full) - train_size
cifar10_train, cifar10_val = random_split(cifar10_full, [train_size, val_size])  # train, val
trainloader = DataLoader(cifar10_train, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
valloader = DataLoader(cifar10_val, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

# test set
cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, transform=custom_torchvision.preprocess, download=True)
testloader = DataLoader(cifar10_test, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())


def train():
    # data
    cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)

    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, transform=custom_torchvision.preprocess, download=True)
    testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=hyperparams["batch_size"], shuffle=False, drop_last=False, num_workers=4, pin_memory=torch.cuda.is_available())

    print(f"loaded data")

    # model
    device = get_device(disable_mps=False)
    net = custom_torchvision.resnet152_ensemble(num_classes=len(cifar10_classes))
    custom_torchvision.set_resnet_weights(net, models.ResNet152_Weights.IMAGENET1K_V1)
    custom_torchvision.freeze_backbone(net)
    net = net.to(device)
    ensemble_size = len(net.fc_layers)
    print(f"loaded model")

    #
    # train loop
    #

    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    # optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams["lr"])
    optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams["lr"], momentum=hyperparams["momentum"], weight_decay=hyperparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams["num_epochs"])

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
        net.train()
        running_losses = [0.0] * ensemble_size
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            losses = training_step(outputs=outputs, labels=labels)

            for i in range(ensemble_size):
                running_losses[i] += losses[i].item()  # accumulate losses

            # print stats
            _, predicted = outputs.max(2)
            total += labels.size(0)
            correct += predicted.eq(labels.unsqueeze(1)).sum().item()
            if batch_idx % 20 == 19:
                avg_loss = sum(running_losses) / len(running_losses) / 20
                accuracy = 100.0 * correct / total
                print(f"[Epoch {epoch + 1} | {batch_idx + 1:5d}/{len(trainloader)}] Avg Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%")
                running_losses = [0.0] * ensemble_size
                correct = 0
                total = 0

        scheduler.step()

        # validation
        net.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs.mean(1), labels)
                val_loss += loss.item()
                _, predicted = outputs.max(2)
                total += labels.size(0)
                correct += predicted.eq(labels.unsqueeze(1)).sum().item()

        val_accuracy = 100.0 * correct / total
        print(f"Validation Loss: {val_loss/len(valloader):.3f}, Accuracy: {val_accuracy:.2f}%")

    # save model
    model_path = Path.cwd() / "data" / "model.pth"
    torch.save(net.state_dict(), model_path)
    print("saved model")


def eval():
    # data
    cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, transform=custom_torchvision.preprocess, download=True)
    testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=hyperparams["batch_size"], shuffle=False, drop_last=False, num_workers=4, pin_memory=torch.cuda.is_available())
    print(f"loaded data")

    # model
    set_seed()
    device = get_device(disable_mps=False)
    model_path = Path.cwd() / "data" / "model.pth"
    net = custom_torchvision.resnet152_ensemble(num_classes=len(cifar10_classes))
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    net = net.to(device)
    print(f"loaded model")

    #
    # eval loop
    #

    correct_last = 0
    correct_mean = 0
    correct_median = 0
    correct_crossmax = 0
    total = len(cifar10_test)

    def cross_max(outputs, k=2, self_assemble_mode=True):
        Z_hat = outputs - outputs.max(dim=2, keepdim=True)[0]  # subtract the max per-predictor over classes
        Z_hat = Z_hat - Z_hat.max(dim=1, keepdim=True)[0]  # subtract the per-class max over predictors
        Y, _ = torch.topk(Z_hat, k, dim=1)  # choose the kth highest logit per class
        if self_assemble_mode:
            Y = torch.median(Z_hat, dim=1)[0]  # get median value
        else:
            Y = Y[:, -1, :]  # get k-th highest value
        return Y

    with torch.no_grad(), torch.inference_mode():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)  # shape: [batch_size, ensemble_size, num_classes]

            _, predicted_last = torch.max(outputs[:, -1, :], 1)
            correct_last += (predicted_last == labels).sum().item()

            _, predicted_mean = torch.max(outputs.mean(dim=1), 1)
            correct_mean += (predicted_mean == labels).sum().item()

            _, predicted_median = torch.max(outputs.median(dim=1)[0], 1)
            correct_median += (predicted_median == labels).sum().item()

            _, predicted_crossmax = torch.max(cross_max(outputs), 1)
            correct_crossmax += (predicted_crossmax == labels).sum().item()

            free_mem()

    results = {
        "hyperparams": hyperparams,
        "acc_last": correct_last / total,
        "acc_mean": correct_mean / total,
        "acc_median": correct_median / total,
        "acc_crossmax": correct_crossmax / total,
    }
    with open(output_path / "hyperparams.json", "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    train()
    # eval()
