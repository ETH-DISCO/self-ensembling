import gc
import itertools
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.amp import GradScaler
from tqdm import tqdm

import custom_torchvision
from dataloader import get_cifar10_loaders, get_cifar100_loaders
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
    custom_torchvision.set_imagenet_backbone(model)
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

    # save model
    tensors = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    save_file(tensors, f"model_{config['dataset']}.safetensors")


if __name__ == "__main__":
    searchspace = {
        "dataset": ["cifar10", "cifar100"],
        "lr": [1e-4],
        "num_epochs": [16],
        "crossmax_k": [2],
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    for combination in combinations:
        train(config=combination)
