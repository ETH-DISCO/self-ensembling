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
from utils import free_mem, get_device, set_env

set_env(seed=41)

input_path = Path.cwd() / "data"
output_path = Path.cwd() / "data" / "hyperparams.jsonl"
dataset_path = Path.cwd() / "dataset"

full_dataset_cifar10 = datasets.CIFAR10(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
full_dataset_cifar100 = datasets.CIFAR100(root=dataset_path, train=True, transform=custom_torchvision.preprocess, download=True)
# full_dataset_imagenet = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="train", streaming=False) # takes ~1h
# full_dataset_imagenet = [(custom_torchvision.preprocess(x["image"].convert("RGB")), x["label"]) for x in tqdm(full_dataset_imagenet)]  # takes ~1h
print("loaded datasets")


def train(config: dict):
    #
    # dataset
    #

    if config["dataset"] == "cifar10":
        classes = json.loads((input_path / "cifar10_classes.json").read_text())

        train_size = int(0.8 * len(full_dataset_cifar10))
        val_size = len(full_dataset_cifar10) - train_size
        train_dataset, val_dataset = random_split(full_dataset_cifar10, [train_size, val_size])

    elif config["dataset"] == "cifar100":
        classes = json.loads((input_path / "cifar100_classes.json").read_text())

        train_size = int(0.8 * len(full_dataset_cifar100))
        val_size = len(full_dataset_cifar100) - train_size
        train_dataset, val_dataset = random_split(full_dataset_cifar100, [train_size, val_size])

    # elif config["dataset"] == "imagenet":
    #     classes = json.loads((input_path / "imagenet_classes.json").read_text())

    #     train_size = int(0.8 * len(full_dataset_imagenet))
    #     val_size = len(full_dataset_imagenet) - train_size
    #     train_dataset, val_dataset = random_split(full_dataset_imagenet, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    #
    # load backbone
    #

    device = get_device(disable_mps=True)
    net = custom_torchvision.resnet152_ensemble(num_classes=len(classes))
    custom_torchvision.set_resnet_weights(net, models.ResNet152_Weights.IMAGENET1K_V1)
    custom_torchvision.freeze_backbone(net)
    net = net.to(device)  # compilation speedup is insignificant, will break on mps

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

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config["num_epochs"]):
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
                print(f"[epoch {epoch + 1}/{config['num_epochs']}: {batch_idx + 1}/{train_size}] ensemble loss: {', '.join(f'{l:.3f}' for l in running_losses)}")
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
        print(f"epoch {epoch + 1}/{config['num_epochs']}, validation Loss: {val_loss:.4f}")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = net.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= config["early_stopping_patience"]:
            print(f"early stopping triggered after {epoch + 1} epochs")
            break

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
    # - https://github.com/weiaicunzai/pytorch-cifar100
    # - https://www.researchgate.net/figure/Hyperparameters-with-optimal-values-for-ResNet-models-on-CIFAR-100-dataset_tbl1_351654208
    #

    searchspace = {
        "dataset": ["cifar10", "cifar100"],
        "batch_size": 32, # lower is better, but slower (usually 32-128)
        "lr": 0.1,
        "num_epochs": 250, # higher is better, but slower (usually 200-300)
        "crossmax_k": 2, # 2 because we assume vickery voting system (this can be varied after training is done)
        "early_stopping_patience": 10, # higher is better, but slower (usually 5-20)
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    print(f"searching {len(combinations)} combinations")
    
    for combination in combinations:
        GREEN = "\033[92m"
        END = "\033[0m"

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

        if is_cached(combination):
            print(f"{GREEN}skipping: {combination}{END}")
            continue

        print(f"{GREEN}training: {combination}{END}")
        train(config=combination)
