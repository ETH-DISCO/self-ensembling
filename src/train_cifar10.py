from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm

import custom_torchvision
from utils import get_device, set_seed


def main():
    #
    # data
    #

    datapath = Path.cwd() / "data"
    batch_size = 256
    num_workers = 4

    cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    cifar10_train = datasets.CIFAR10(root=datapath, train=True, transform=custom_torchvision.preprocess, download=True)
    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    cifar10_test = datasets.CIFAR10(root=datapath, train=False, transform=custom_torchvision.preprocess, download=True)
    testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    print(f"loaded data")

    #
    # model
    #

    set_seed()
    device = get_device(disable_mps=False)
    print(f"device: {device}")

    net = custom_torchvision.resnet152_ensemble(num_classes=len(cifar10_classes))
    custom_torchvision.set_resnet_weights(net, models.ResNet152_Weights.IMAGENET1K_V1)
    custom_torchvision.freeze_backbone(net)

    net = net.to(device)
    ensemble_size = len(net.fc_layers)
    print(f"loaded model")

    #
    # train loop
    #

    lr = 1e-4
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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

    for epoch in range(2):  # loop over full dataset twice
        running_losses = [0.0] * ensemble_size
        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            losses = training_step(outputs=outputs, labels=labels)
            for i in range(ensemble_size):
                running_losses[i] += losses[i].item()

            if batch_idx % 20 == 19:
                print(f"[epoch {epoch + 1} | {batch_idx + 1:5d}/{len(trainloader)}] losses: {', '.join(f'{l:.3f}' for l in running_losses)}")
                running_losses = [0.0] * ensemble_size

    #
    # save model
    #

    model_path = Path.cwd() / "data" / "model.pth"
    torch.save(net.state_dict(), model_path)
    print("saved model")


if __name__ == "__main__":
    main()
