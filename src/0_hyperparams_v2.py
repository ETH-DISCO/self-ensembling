"""
improvement: single gpu implementation, without early stopping and lots of memory usage optimizations
"""

import gc
import itertools
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import GradScaler
from tqdm import tqdm

import custom_torchvision
from dataloader import get_cifar10_loaders, get_cifar100_loaders, get_resnet152_imagenet_weights
from utils import print_gpu_memory, set_env

set_env(seed=41)
assert torch.cuda.is_available(), "cuda is not available"

output_path = Path.cwd() / "data" / "hyperparams.jsonl"

batch_size = 8
gradient_accumulation_steps = 8
train_val_ratio = 0.8

cifar10_classes, cifar10_trainloader, cifar10_valloader, cifar10_testloader = get_cifar10_loaders(batch_size, train_val_ratio)
cifar100_classes, cifar100_trainloader, cifar100_valloader, cifar100_testloader = get_cifar100_loaders(batch_size, train_val_ratio)


def train(config: dict):
    if config["dataset"] == "cifar10":
        classes, trainloader, valloader = cifar10_classes, cifar10_trainloader, cifar10_valloader
    elif config["dataset"] == "cifar100":
        classes, trainloader, valloader = cifar100_classes, cifar100_trainloader, cifar100_valloader

    device = torch.device("cuda")
    model = custom_torchvision.get_custom_resnet152(num_classes=len(classes)).to(device)
    weights = get_resnet152_imagenet_weights()
    custom_torchvision.set_backbone_weights(model, weights)
    custom_torchvision.freeze_backbone(model)
    model.use_checkpoint = True
    model = torch.compile(model, mode="reduce-overhead")

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], fused=True)
    scaler = GradScaler(device="cuda", enabled=True)
    ensemble_size = len(model.fc_layers)
    train_size = len(trainloader)

    for epoch in range(config["num_epochs"]):
        model.train()
        running_losses = torch.zeros(ensemble_size, device=device)

        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=train_size):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(inputs)
                losses = torch.stack([criterion(outputs[:, i, :], labels) for i in range(ensemble_size)])
                total_loss = losses.sum()

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_losses += losses.detach()

            if batch_idx % (train_size // 3) == 0:
                print(f"[epoch {epoch + 1}/{config['num_epochs']}: {batch_idx + 1}/{train_size}] ensemble loss: {', '.join(f'{l:.3f}' for l in running_losses)}")
                running_losses.zero_()

        gc.collect()
        torch.cuda.empty_cache()
        if (epoch + 1) % 5 == 0:
            print_gpu_memory()

    # validation
    y_true, y_preds = [], []
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=True):
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_preds.extend(custom_torchvision.get_cross_max_consensus(outputs=outputs, k=config["crossmax_k"]).cpu().numpy())
    results = {
        "config": config,
        "accuracy": accuracy_score(y_true, y_preds),
        "precision": precision_score(y_true, y_preds, average="weighted"),
        "recall": recall_score(y_true, y_preds, average="weighted"),
        "f1_score": f1_score(y_true, y_preds, average="weighted"),
    }
    print(f"validation accuracy: {results['accuracy']:.3f}")
    with open(output_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    # save model
    # from safetensors import save_file
    # model_state_dict = model.state_dict()
    # tensors = {k: v.cpu() for k, v in model_state_dict.items()}
    # save_file(tensors, f"model_{config['dataset']}_{config['num_epochs']}epochs.safetensors")


if __name__ == "__main__":

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
        "lr": [1e-1, 1e-4, 1e-5, 1.7e-5, 1e-6, 1e-7],  # paper found 1.7e-5 to be most robust
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
