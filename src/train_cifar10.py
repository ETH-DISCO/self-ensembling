import json
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import custom_torchvision
from utils import free_mem, get_device, set_seed

set_seed()

#
# hyperparams
#

hyperparams = {
    "batch_size": 256,
    "lr": 1e-4,
    "num_epochs": 2,
    "crossmax_k": 2,
}

#
# data
#

dataset_path = Path.cwd() / "dataset"

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
    # data
    cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
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
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams["lr"])  # see: https://karpathy.github.io/2019/04/25/recipe/

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
                running_losses[i] += losses[i].item()  # accumulate losses

            if batch_idx % 20 == 19:
                print(f"[epoch {epoch + 1} | {batch_idx + 1:5d}/{len(trainloader)}] losses: {', '.join(f'{l:.3f}' for l in running_losses)}")
                running_losses = [0.0] * ensemble_size

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

    correct_crossmax = 0
    total = len(cifar10_test)
    with torch.no_grad(), torch.inference_mode():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)  # shape: [batch_size, ensemble_size, num_classes]

            predictions = custom_torchvision.get_cross_max_consensus(outputs=outputs, k=hyperparams["crossmax_k"])
            correct_crossmax += (predictions == labels).sum().item()

            free_mem()

    output_path = Path.cwd() / "data"
    results = {
        "hyperparams": hyperparams,
        "acc_crossmax": correct_crossmax / total,
    }
    with open(output_path / "hyperparams.json", "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    train()
    eval()
