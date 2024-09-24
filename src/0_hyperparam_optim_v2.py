import itertools
import json
from pathlib import Path

import torch
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import custom_torchvision
from dataloader import get_cifar10_loaders, get_cifar100_loaders
from utils import free_mem, get_device, set_env

set_env(seed=41)

output_path = Path.cwd() / "data" / "hyperparams.jsonl"

#
# config constants
#

batch_size = 1024  # lower always better, but slower
train_ratio = 0.8  # common default
num_epochs = 250  # higher with early stopping is better, but slower (usually 200-300)
early_stopping_patience = 10  # higher is better, but slower (usually 5-20)

cifar10_classes, cifar10_trainloader, cifar10_valloader, cifar10_testloader = get_cifar10_loaders(batch_size, train_ratio)
cifar100_classes, cifar100_trainloader, cifar100_valloader, cifar100_testloader = get_cifar100_loaders(batch_size, train_ratio)


def train(config: dict):
    if config["dataset"] == "cifar10":
        classes = cifar10_classes
        trainloader = cifar10_trainloader
        valloader = cifar10_valloader
    elif config["dataset"] == "cifar100":
        classes = cifar100_classes
        trainloader = cifar100_trainloader
        valloader = cifar100_valloader

    device = get_device(disable_mps=False)
    net = custom_torchvision.resnet152_ensemble(num_classes=len(classes))
    custom_torchvision.set_resnet_weights(net, models.ResNet152_Weights.IMAGENET1K_V1) # use imagenet weights
    custom_torchvision.freeze_backbone(net)
    net = net.to(device)  # dont compile: speedup is insignificant, will break on mps

    #
    # train loop
    #

    criterion = torch.nn.CrossEntropyLoss().to(device)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])  # safe bet
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

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # epoch train
        net.train()
        running_losses = [0.0] * ensemble_size
        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=train_size):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            losses = training_step(outputs=outputs, labels=labels)
            for i in range(ensemble_size):
                running_losses[i] += losses[i].item()

            # print stats
            if batch_idx % (train_size // 3) == 0:
                print(f"[epoch {epoch + 1}/{num_epochs}: {batch_idx + 1}/{train_size}] ensemble loss: {', '.join(f'{l:.3f}' for l in running_losses)}")
                running_losses = [0.0] * ensemble_size
            free_mem()

        # epoch validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=("cuda" in str(device))), torch.inference_mode():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = sum(criterion(outputs[:, i, :], labels) for i in range(ensemble_size))
                val_loss += loss.item()

        val_loss /= len(valloader)
        print(f"epoch {epoch + 1}/{num_epochs}, validation Loss: {val_loss:.4f}")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = net.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"early stopping triggered after {epoch + 1} epochs")
            break
        free_mem()

    #
    # validation loop
    #

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    y_true = []
    y_pred = []

    net.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=("cuda" in str(device))), torch.inference_mode():
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
    with open(output_path, "a") as f:
        f.write(json.dumps(results) + "\n")


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
        "dataset": ["cifar10", "cifar100"],
        "lr": [1e-2, 1e-3, 1e-4],
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
