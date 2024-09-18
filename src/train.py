from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm

import custom_torchvision
from utils import free_mem, get_device, set_seed, timeit

hyperparams = {
    "batch_size": 256,
    "num_workers": 4,
    "lr": 1e-4,
    "num_epochs": 2,
}


@timeit
def train():
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


def cross_max(outputs, k=2, self_assemble_mode=True):
    # logits Z of shape [B, N, C]
    # - B: batch size
    # - N: number of predictors
    # - C: number of classes

    # subtract the max per-predictor over classes
    Z_hat = outputs - outputs.max(dim=2, keepdim=True)[0]

    # subtract the per-class max over predictors
    Z_hat = Z_hat - Z_hat.max(dim=1, keepdim=True)[0]

    # choose the kth highest logit per class
    Y, _ = torch.topk(Z_hat, k, dim=1)

    if self_assemble_mode:
        Y = torch.median(Z_hat, dim=1)[0]  # get median value
    else:
        Y = Y[:, -1, :]  # get k-th highest value

    return Y


@timeit
def eval():
    #
    # data
    #

    datapath = Path.cwd() / "data"
    batch_size = 256
    num_workers = 4

    cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    cifar10_test = datasets.CIFAR10(root=datapath, train=False, transform=custom_torchvision.preprocess, download=True)
    testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    print(f"loaded data")

    #
    # model
    #

    set_seed()
    device = get_device(disable_mps=False)
    print(f"device: {device}")

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

    with torch.no_grad(), torch.inference_mode():
        for data in tqdm(testloader):
            images, labels = data
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

    print("-" * 20)
    print(f"hyperparams: {hyperparams}")
    print("-" * 20)
    print(f"acc (last layer): {correct_last / total:.2%}")
    print(f"acc (mean self-ensemble): {correct_mean / total:.2%}")
    print(f"acc (median self-ensemble): {correct_median / total:.2%}")
    print(f"acc (crossmax self-ensemble): {correct_crossmax / total:.2%}")


if __name__ == "__main__":
    train()
    eval()
