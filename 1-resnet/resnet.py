import hashlib
import json
import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm
from utils import *

assert torch.cuda.is_available()
set_env()

data_path = get_current_dir().parent / "data"
dataset_path = get_current_dir().parent / "datasets"
weights_path = get_current_dir().parent / "weights"
output_path = get_current_dir()
mask_path = get_current_dir() / "masks"

os.makedirs(data_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)


def get_dataset(dataset: str):
    if dataset == "cifar10":
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
        original_images_train_np = np.array(trainset.data)
        original_labels_train_np = np.array(trainset.targets)
        original_images_test_np = np.array(testset.data)
        original_labels_test_np = np.array(testset.targets)
    elif dataset == "cifar100":
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
        original_images_train_np = np.array(trainset.data)
        original_labels_train_np = np.array(trainset.targets)
        original_images_test_np = np.array(testset.data)
        original_labels_test_np = np.array(testset.targets)
    elif dataset == "imagenette":
        num_classes = 10
        imgpath = dataset_path / "imagenette2"
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
        trainset = torchvision.datasets.Imagenette(root=dataset_path, split="train", download=(not (imgpath / "train").exists()), transform=transform)
        testset = torchvision.datasets.Imagenette(root=dataset_path, split="val", download=(not (imgpath / "val").exists()), transform=transform)
        num_train = len(trainset)
        num_test = len(testset)
        original_images_train_np = np.empty((num_train, 224, 224, 3), dtype=np.uint8)
        original_labels_train_np = np.empty(num_train, dtype=np.int64)
        original_images_test_np = np.empty((num_test, 224, 224, 3), dtype=np.uint8)
        original_labels_test_np = np.empty(num_test, dtype=np.int64)
        for idx, (image, label) in tqdm(enumerate(trainset), total=num_train, desc="preprocessing trainset", ncols=100):
            original_images_train_np[idx] = (np.transpose(image.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            original_labels_train_np[idx] = label
        for idx, (image, label) in tqdm(enumerate(testset), total=num_test, desc="preprocessing testset", ncols=100):
            original_images_test_np[idx] = (np.transpose(image.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            original_labels_test_np[idx] = label
    else:
        assert False

    images_train_np = original_images_train_np / 255.0
    images_test_np = original_images_test_np / 255.0
    labels_train_np = original_labels_train_np
    labels_test_np = original_labels_test_np
    return images_train_np, labels_train_np, images_test_np, labels_test_np, num_classes


def apply_hcaptcha_mask(images, opacity: int, mask_sides=3, mask_per_rowcol=2, mask_num_concentric=2, mask_colors=True):
    def add_overlay(background: Image.Image, overlay: Image.Image, opacity: int) -> Image.Image:
        overlay = overlay.resize(background.size)
        result = Image.new("RGBA", background.size)
        result.paste(background, (0, 0))
        mask = Image.new("L", overlay.size, opacity)  # 0=transparent; 255=opaque
        result.paste(overlay, (0, 0), mask)
        return result

    mask = Image.open(mask_path / f"{mask_sides}_{mask_per_rowcol}_{mask_num_concentric}_{mask_colors}.png")

    all_perturbed_images = []
    to_pil = lambda x: Image.fromarray((x * 255).astype(np.uint8))
    to_np = lambda x: np.array(x) / 255.0
    for i in tqdm(range(len(images)), desc="adding overlay", ncols=100):
        perturbed_image = to_np(add_overlay(to_pil(images[i]), mask, opacity).convert("RGB"))
        all_perturbed_images.append(perturbed_image)
    return np.array(all_perturbed_images)


def get_model(
    dataset,  # used by cache hash
    num_classes,
    images_train_np,
    labels_train_np,
    images_test_np,
    labels_test_np,
    # train config
    train_num_epochs=0,
    train_hcaptcha_ratio=0.0,
    train_hcaptcha_opacity=0,
    # mask config
    mask_sides=3,
    mask_per_rowcol=2,
    mask_num_concentric=2,
    mask_colors=True,
):
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, num_classes)
    model = model.to("cuda")
    model.train()
    free_mem()

    if train_num_epochs == 0:
        return model

    args_hash = hashlib.md5(json.dumps({k: v for k, v in locals().items() if isinstance(v, (int, float, str, bool, list, dict))}, sort_keys=True).encode()).hexdigest()
    cache_name = f"tmp_{args_hash}.pth"
    if (weights_path / cache_name).exists():
        model.load_state_dict(torch.load(weights_path / cache_name))
        print(f"loaded cached model: {cache_name}")
        return model

    if train_hcaptcha_ratio > 0.0:
        num_total = len(labels_train_np)
        num_perturbed = int(train_hcaptcha_ratio * num_total)
        perturbed_indices = np.random.choice(num_total, num_perturbed, replace=False)
        images_train_np[perturbed_indices] = apply_hcaptcha_mask(images_train_np[perturbed_indices], opacity=train_hcaptcha_opacity, mask_sides=mask_sides, mask_per_rowcol=mask_per_rowcol, mask_num_concentric=mask_num_concentric, mask_colors=mask_colors)

    learning_rate = 0.001
    batch_size = 128
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(torch.FloatTensor(images_train_np).permute(0, 3, 1, 2), torch.LongTensor(labels_train_np))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.FloatTensor(images_test_np).permute(0, 3, 1, 2), torch.LongTensor(labels_test_np))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(train_num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{train_num_epochs}", ncols=100):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * correct / total
        print(f"epoch [{epoch+1}/{train_num_epochs}]: train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%, test loss: {test_loss:.4f}, test acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), weights_path / cache_name)
    print(f"cached model {cache_name} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f} MB)")
    return model


def eval_model(
    model,
    images_test_np,
    labels_test_np,
):
    model.eval()
    test_dataset = TensorDataset(torch.FloatTensor(images_test_np).permute(0, 3, 1, 2), torch.LongTensor(labels_test_np))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="evaluating", ncols=100):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def is_cached(filepath, combination):
    if filepath.exists() and filepath.stat().st_size > 0:
        lines = filepath.read_text().strip().split("\n")
        lines = [json.loads(line) for line in lines]
        for line in lines:
            if all(line[k] == v for k, v in combination.items()):
                return True
    return False


if __name__ == "__main__":
    fpath = output_path / "resnet.jsonl"
    fpath.touch(exist_ok=True)

    combinations = {
        "dataset": ["cifar10", "cifar100", "imagenette"],
        # train config
        "train_epochs": [0, 2, 6],
        "train_hcaptcha_ratio": [0.0, 0.5, 1.0],
        "train_opacity": [0, 2, 4, 8, 16, 32, 64, 128],
        # mask config
        "mask_sides": [3, 4, 6, 10],
        "mask_per_rowcol": [2, 4, 10],
        "mask_num_concentric": [2, 5, 10],
        "mask_colors": [True, False],
    }
    combs = list(product(*combinations.values()))
    print(f"total combinations: {len(combs)}")

    for idx, comb in enumerate(combs):
        print(f"progress: {idx+1}/{len(combs)}")
        comb = {k: v for k, v in zip(combinations.keys(), comb)}
        if is_cached(fpath, comb):
            continue

        images_train_np, labels_train_np, images_test_np, labels_test_np, num_classes = get_dataset(comb["dataset"])

        model = get_model(
            comb["dataset"],
            num_classes,
            images_train_np.copy(),
            labels_train_np.copy(),
            images_test_np.copy(),
            labels_test_np.copy(),
            # train config
            train_num_epochs=comb["train_epochs"],
            train_hcaptcha_ratio=comb["train_hcaptcha_ratio"],
            train_hcaptcha_opacity=comb["train_opacity"],
            # mask config
            mask_sides=comb["mask_sides"],
            mask_per_rowcol=comb["mask_per_rowcol"],
            mask_num_concentric=comb["mask_num_concentric"],
            mask_colors=comb["mask_colors"],
        )

        output = {
            **comb,
            "acc": eval_model(model, images_test_np.copy(), labels_test_np.copy()),
        }
 
        # eval_opacities = [0, 2, 4, 8, 16, 32, 64, 128]
        # for opacity in eval_opacities:
        #     output[f"acc_{opacity}"] = eval_model(
        #         model,
        #         apply_hcaptcha_mask(images_test_np.copy(), opacity=opacity, mask_sides=comb["mask_sides"], mask_per_rowcol=comb["mask_per_rowcol"], mask_num_concentric=comb["mask_num_concentric"], mask_colors=comb["mask_colors"]),
        #         labels_test_np.copy(),
        #     )

        mask_opacities = [0, 1, 2, 4, 8, 16, 32, 64, 128]
        mask_sides = [3, 4, 6, 10]
        mask_per_rowcols = [2, 4, 10]
        mask_num_concentrics = [2, 5, 10]
        mask_colors = [True, False]
        for opacity in mask_opacities:
            for side in mask_sides:
                for per_rowcol in mask_per_rowcols:
                    for num_concentric in mask_num_concentrics:
                        for colors in mask_colors:
                            output[f"acc_{opacity}_{side}_{per_rowcol}_{num_concentric}_{colors}"] = eval_model(
                                model,
                                apply_hcaptcha_mask(images_test_np.copy(), opacity=opacity, mask_sides=side, mask_per_rowcol=per_rowcol, mask_num_concentric=num_concentric, mask_colors=colors),
                                labels_test_np.copy(),
                            )

        with fpath.open("a") as f:
            f.write(json.dumps(output) + "\n")
