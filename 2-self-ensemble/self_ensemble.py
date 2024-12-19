import copy
import hashlib
import json
import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from tqdm import tqdm
from utils import *

assert torch.cuda.is_available()
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_memory >= 79, "at least 80GB gpu memory required"
set_env()

data_path = get_current_dir().parent / "data"
dataset_path = get_current_dir().parent / "datasets"
weights_path = get_current_dir().parent / "weights"
output_path = get_current_dir()

os.makedirs(data_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

prerendered_mask = Image.open(get_current_dir() / "masks" / "mask.png")


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


"""
training phase
"""


def make_multichannel_input(images, enable_noise, enable_random_shuffle, resolutions):
    down_noise = 0.2  # noise standard deviation to be added at the low resolution
    up_noise = 0.2  # noise stadard deviation to be added at the high resolution
    jit_size = 3  # max size of the x-y jit in each axis, sampled uniformly from -jit_size to +jit_size inclusive
    up_res = 32  # hard coded for CIFAR-10 or CIFAR-100  <---------------- THIS IS WHY IMAGENETTE ISN'T WORKING

    if not enable_noise:
        return torch.concatenate([images] * len(resolutions), axis=1)  # don't do anything

    all_channels = []
    for i, r in enumerate(resolutions):

        def apply_transformations(images, down_res, up_res, jit_x, jit_y, down_noise, up_noise, contrast, color_amount):
            # images = torch.mean(images, axis=1, keepdims=True) # only for mnist
            images_collected = []
            for i in range(images.shape[0]):
                image = images[i]
                image = torchvision.transforms.functional.adjust_contrast(image, contrast[i])  # changing contrast
                image = torch.roll(image, shifts=(jit_x[i], jit_y[i]), dims=(-2, -1))  # shift the result in x and y
                image = color_amount[i] * image + torch.mean(image, axis=0, keepdims=True) * (1 - color_amount[i])  # shifting in the color <-> grayscale axis
                images_collected.append(image)
            images = torch.stack(images_collected, axis=0)
            images = F.interpolate(images, size=(down_res, down_res), mode="bicubic")  # descrease the resolution
            noise = down_noise * torch.Tensor(np.random.rand(images.shape[0], 3, down_res, down_res)).to("cuda")  # low res noise
            images = images + noise
            images = F.interpolate(images, size=(up_res, up_res), mode="bicubic")  # increase the resolution
            noise = up_noise * torch.Tensor(np.random.rand(images.shape[0], 3, up_res, up_res)).to("cuda")  # high res noise
            images = images + noise
            images = torch.clip(images, 0, 1)  # clipping to the right range of values
            return images

        images_now = apply_transformations(
            images,
            down_res=r,
            up_res=up_res,
            jit_x=np.random.choice(range(-jit_size, jit_size + 1), len(images + i)),  # x-shift,
            jit_y=np.random.choice(range(-jit_size, jit_size + 1), len(51 * images + 7 * i + 125 * r)),  # y-shift
            down_noise=down_noise,
            up_noise=up_noise,
            contrast=np.random.choice(np.linspace(0.7, 1.5, 100), len(7 + 3 * images + 9 * i + 5 * r)),  # change in contrast,
            color_amount=np.random.choice(np.linspace(0.5, 1.0, 100), len(5 + 7 * images + 8 * i + 2 * r)),  # change in color amount
        )
        all_channels.append(images_now)

    if not enable_random_shuffle:
        return torch.concatenate(all_channels, axis=1)
    elif enable_random_shuffle:
        indices = torch.randperm(len(all_channels))
        shuffled_tensor_list = [all_channels[i] for i in indices]
        return torch.concatenate(shuffled_tensor_list, axis=1)


def eval_model(model, images_in, labels_in, batch_size=128):
    all_preds = []
    all_logits = []

    with torch.no_grad():
        its = int(np.ceil(float(len(images_in)) / float(batch_size)))
        pbar = tqdm(range(its), desc="eval", ncols=100)
        for it in pbar:
            i1 = it * batch_size
            i2 = min([(it + 1) * batch_size, len(images_in)])

            inputs = torch.Tensor(images_in[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
            outputs = model(inputs)

            all_logits.append(outputs.detach().cpu().numpy())
            preds = torch.argmax(outputs, axis=-1)
            all_preds.append(preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    return np.sum(all_preds == labels_in), all_preds.shape[0], all_logits  # hits, count, logits


def train_model(model_in, images_in, labels_in, epochs, lr, batch_size, optimizer_in, subset_only, mode, use_adversarial_training, adversarial_epsilon, skip_test_set_eval):
    def fgsm_attack(model, xs, ys, epsilon, random_reps, batch_size):
        model = model.eval()
        model = model.cuda()

        all_perturbed_images = []

        its = int(np.ceil(xs.shape[0] / batch_size))
        for it in range(its):
            i1 = it * batch_size
            i2 = min([(it + 1) * batch_size, xs.shape[0]])

            x = torch.Tensor(xs[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
            y = torch.Tensor(ys[i1:i2]).to("cuda").to(torch.long)
            x.requires_grad = True

            for _ in range(random_reps):
                outputs = model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)
                loss.backward()

            perturbed_image = x + epsilon * x.grad.data.sign()
            perturbed_image = torch.clip(perturbed_image, 0, 1)
            all_perturbed_images.append(perturbed_image.detach().cpu().numpy().transpose([0, 2, 3, 1]))
        return np.concatenate(all_perturbed_images, axis=0)

    global storing_models

    train_accs = []
    test_accs = []

    if mode == "train":
        model_in.train()
    elif mode == "eval":
        model_in.eval()

    criterion = nn.CrossEntropyLoss()

    if subset_only is None:
        train_optimizer = optimizer_in(model_in.parameters(), lr=lr)
    else:
        train_optimizer = optimizer_in(subset_only, lr=lr)

    for epoch in range(epochs):
        print(f"epoch={epoch}/{epochs}")
        if mode == "train":  # avoid random flips
            model_in.train()
        elif mode == "eval":
            model_in.eval()
        else:
            assert False

        all_hits = []

        randomized_ids = np.random.permutation(range(len(images_in)))
        its = int(np.ceil(float(len(images_in)) / float(batch_size)))
        pbar = tqdm(range(its), desc="Training", ncols=100)
        for it in pbar:
            i1 = it * batch_size
            i2 = min([(it + 1) * batch_size, len(images_in)])

            np_images_used = images_in[randomized_ids[i1:i2]]
            np_labels_used = labels_in[randomized_ids[i1:i2]]

            if use_adversarial_training:  # very light adversarial training if on
                attacked_images = fgsm_attack(model_in.eval(), np_images_used[:], np_labels_used[:], epsilon=adversarial_epsilon, random_reps=1, batch_size=batch_size // 2)
                np_images_used = attacked_images
                np_labels_used = np_labels_used
                if mode == "train":
                    model_in.train()
                elif mode == "eval":
                    model_in.eval()

            inputs = torch.Tensor(np_images_used.transpose([0, 3, 1, 2])).to("cuda")
            labels = torch.Tensor(np_labels_used).to("cuda").to(torch.long)

            # the actual optimization step
            train_optimizer.zero_grad()
            outputs = model_in(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_optimizer.step()

            # on the fly eval for the train set batches
            preds = torch.argmax(outputs, axis=-1)
            acc = torch.mean((preds == labels).to(torch.float), axis=-1)
            all_hits.append((preds == labels).to(torch.float).detach().cpu().numpy())
            train_accs.append(acc.detach().cpu().numpy())

            pbar.set_description(f"train acc={acc.detach().cpu().numpy()} loss={loss.item()}")

        if not skip_test_set_eval:
            with isolated_environment():
                eval_model_copy = copy.deepcopy(model_in)
                test_hits, test_count, _ = eval_model(eval_model_copy.eval(), images_test_np, labels_test_np)

        # end of epoch eval
        train_hits = np.sum(np.concatenate(all_hits, axis=0).reshape([-1]))
        train_count = np.concatenate(all_hits, axis=0).reshape([-1]).shape[0]
        print(f"e={epoch} train {train_hits} / {train_count} = {train_hits/train_count},  test {test_hits} / {test_count} = {test_hits/test_count}")

        test_accs.append((test_hits / test_count) if (test_count > 0) else 0)
    return model_in, train_accs, test_accs


class WrapModelForResNet152(torch.nn.Module):
    def __init__(self, model, multichannel_fn, num_classes=10):
        super(WrapModelForResNet152, self).__init__()
        self.multichannel_fn = multichannel_fn  # multi-res input
        self.model = model
        self.num_classes = num_classes
        self.layer_operations = [
            torch.nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
            ),
            *model.layer1,
            *model.layer2,
            *model.layer3,
            *model.layer4,
            model.avgpool,
            model.fc,
        ]
        self.all_dims = [
            3 * 224 * 224 * len(resolutions),
            64 * 56 * 56,
            *[256 * 56 * 56] * len(model.layer1),
            *[512 * 28 * 28] * len(model.layer2),
            *[1024 * 14 * 14] * len(model.layer3),
            *[2048 * 7 * 7] * len(model.layer4),
            2048,
            1000,
        ]

        class BatchNormLinear(nn.Module):
            def __init__(self, in_features, out_features, device="cuda"):
                super(BatchNormLinear, self).__init__()
                self.batch_norm = nn.BatchNorm1d(in_features, device=device)
                self.linear = nn.Linear(in_features, out_features, device=device)

            def forward(self, x):
                x = self.batch_norm(x)
                return self.linear(x)

        self.linear_layers = torch.nn.ModuleList([BatchNormLinear(self.all_dims[i], num_classes, device="cuda") for i in range(len(self.all_dims))])

    def prepare_input(self, x):  # preprocess
        x = self.multichannel_fn(x)
        x = F.interpolate(x, size=(224, 224), mode="bicubic")
        x = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406] * (x.shape[1] // 3), std=[0.229, 0.224, 0.225] * (x.shape[1] // 3))(x)
        return x

    def forward_until(self, x, layer_id):
        x = self.prepare_input(x)
        for l in range(layer_id):
            if list(x.shape)[1:] == [2048, 1, 1]:
                x = x.reshape([-1, 2048])
            x = self.layer_operations[l](x)
        return x

    def forward_after(self, x, layer_id):
        x = self.prepare_input(x)
        for l in range(layer_id, len(self.layer_operations)):
            if list(x.shape)[1:] == [2048, 1, 1]:
                x = x.reshape([-1, 2048])
            x = self.layer_operations[l](x)
        return x

    def predict_from_layer(self, x, l):  # called by LinearNet forward
        x = self.forward_until(x, l)
        x = x.reshape([x.shape[0], -1])
        return self.linear_layers[l](x)

    def predict_from_several_layers(self, x, layers):
        x = self.prepare_input(x)
        outputs = dict()
        outputs[0] = self.linear_layers[0](x.reshape([x.shape[0], -1]))
        for l in range(len(self.layer_operations)):
            if list(x.shape)[1:] == [2048, 1, 1]:
                x = x.reshape([-1, 2048])
            x = self.layer_operations[l](x)
            if l in layers:
                outputs[l + 1] = self.linear_layers[l + 1](x.reshape([x.shape[0], -1]))
        return outputs


class LinearNet(nn.Module):
    def __init__(self, model, layer_i):
        super(LinearNet, self).__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, inputs):
        return self.model.predict_from_layer(inputs, self.layer_i)


def get_model(enable_noise, enable_random_shuffle, enable_adversarial_training, resolutions, layers_to_use, num_classes, images_train_np, labels_train_np):
    #
    # backbone model
    #

    from torchvision.models import ResNet152_Weights, resnet152

    imported_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

    # replace first conv layer with multi-res one
    in_planes = 3
    planes = 64
    stride = 2
    N = len(resolutions)  # input channels multiplier due to multi-res input
    conv2 = nn.Conv2d(in_channels=N * in_planes, out_channels=planes, kernel_size=7, stride=stride, padding=3, bias=False)
    imported_model.conv1 = copy.deepcopy(conv2)

    # set num of classes in final layer
    imported_model.fc = nn.Linear(2048, num_classes)

    class ImportedModelWrapper(nn.Module):
        def __init__(self, imported_model, multichannel_fn):
            super(ImportedModelWrapper, self).__init__()
            self.imported_model = imported_model
            self.multichannel_fn = multichannel_fn

        def forward(self, x):
            # multichannel input
            x = self.multichannel_fn(x)
            # imagenet preprocessing
            x = F.interpolate(x, size=(224, 224), mode="bicubic")
            x = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406] * (x.shape[1] // 3), std=[0.229, 0.224, 0.225] * (x.shape[1] // 3))(x)
            x = self.imported_model(x)
            return x

    configured_make_multichannel_input = lambda x: make_multichannel_input(x, enable_noise=enable_noise, enable_random_shuffle=enable_random_shuffle, resolutions=resolutions)
    wrapped_model = ImportedModelWrapper(imported_model, configured_make_multichannel_input).to("cuda")
    wrapped_model.multichannel_fn = configured_make_multichannel_input
    model = copy.deepcopy(wrapped_model)
    model.multichannel_fn = configured_make_multichannel_input
    model.train()

    # check cache
    args_hash = hashlib.md5(json.dumps({k: v for k, v in locals().items() if isinstance(v, (int, float, str, bool, list, dict))}, sort_keys=True).encode()).hexdigest()
    cache_name = f"tmp_{args_hash}.pth"
    if (weights_path / cache_name).exists():
        cached_model = WrapModelForResNet152(imported_model, configured_make_multichannel_input, num_classes=num_classes)
        cached_model.load_state_dict(torch.load(weights_path / cache_name, weights_only=True))
        print(f"loaded cached model: {cache_name}")
        return cached_model

    torch.cuda.empty_cache()
    model, train_accs, test_accs = train_model(
        model_in=model,
        images_in=images_train_np,
        labels_in=labels_train_np,
        epochs=6,
        lr=3.3e-5,
        batch_size=128,
        optimizer_in=optim.Adam,
        subset_only=None,
        mode="train",
        use_adversarial_training=enable_adversarial_training,
        adversarial_epsilon=8 / 255,
        skip_test_set_eval=False,
    )
    print(f"trained backbone model - avg train acc: {np.mean(train_accs)}, avg test acc: {np.mean(test_accs)}")

    #
    # training linear layers of self-ensemble
    #

    resnet152_wrapper = WrapModelForResNet152(model.imported_model, configured_make_multichannel_input, num_classes=num_classes)
    resnet152_wrapper.multichannel_fn = configured_make_multichannel_input
    resnet152_wrapper = resnet152_wrapper.to("cuda")
    backbone_model = copy.deepcopy(resnet152_wrapper)
    del resnet152_wrapper

    backbone_model.eval()
    linear_model = LinearNet(backbone_model, 5).to("cuda")  # preallocate
    linear_layers_collected_dict = dict()

    for layer_i in reversed(layers_to_use):
        linear_model.layer_i = layer_i
        linear_model.fixed_mode = "train"

        torch.cuda.empty_cache()
        linear_model, train_accs, test_accs = train_model(
            model_in=linear_model,
            images_in=images_train_np[:],
            labels_in=labels_train_np[:],
            epochs=1,
            lr=3.3e-5,
            batch_size=64,
            optimizer_in=optim.Adam,
            subset_only=linear_model.model.linear_layers[layer_i].parameters(),  # just the linear projection
            mode="train",
            use_adversarial_training=False,
            adversarial_epsilon=8 / 255,
            skip_test_set_eval=False,
        )
        linear_layers_collected_dict[layer_i] = copy.deepcopy(backbone_model.linear_layers[layer_i])
        print(f"trained layer={layer_i} - avg train acc: {np.mean(train_accs)}, avg test acc: {np.mean(test_accs)}")

    # copy dict back to backbone
    for layer_i in layers_to_use:
        backbone_model.linear_layers[layer_i] = copy.deepcopy(linear_layers_collected_dict[layer_i])
    del linear_layers_collected_dict

    torch.save(backbone_model.state_dict(), weights_path / cache_name)
    print(f"cached model {cache_name} ({sum(p.numel() for p in backbone_model.parameters()) / 1e6:.2f} MB)")
    return backbone_model


"""
eval phase
"""


def fgsm_attack_layer(model, xs, ys, epsilon, layer_i, batch_size=128):
    model = model.eval()
    model = model.cuda()

    all_perturbed_images = []
    its = int(np.ceil(xs.shape[0] / batch_size))

    for it in range(its):
        i1 = it * batch_size
        i2 = min([(it + 1) * batch_size, xs.shape[0]])

        x = torch.Tensor(xs[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
        y = torch.Tensor(ys[i1:i2]).to("cuda").to(torch.long)
        x.requires_grad = True

        # mod: dropped the random_reps
        layer_output = model.forward_until(x, layer_i)
        layer_logits = model.linear_layers[layer_i](layer_output.reshape(layer_output.shape[0], -1))
        loss = nn.CrossEntropyLoss()(layer_logits, y)
        loss.backward()

        perturbed_image = x + epsilon * x.grad.data.sign()
        perturbed_image = torch.clip(perturbed_image, 0, 1)
        all_perturbed_images.append(perturbed_image.detach().cpu().numpy().transpose([0, 2, 3, 1]))
    return np.concatenate(all_perturbed_images, axis=0)


def pgd_attack_layer(model, xs, ys, epsilon, layer_i, alpha=0.01, num_iter=10, batch_size=128, momentum=0.9, grad_norm_type='sign'):
    # pgd = projected gradient descent
    model = model.eval()
    model = model.cuda()
    
    all_perturbed_images = []
    its = int(np.ceil(xs.shape[0] / batch_size))
    momentum_buffer = None

    for it in tqdm(range(its), desc="pgd attack layer", ncols=100):
        i1 = it * batch_size
        i2 = min([(it + 1) * batch_size, xs.shape[0]])
        current_batch_size = i2 - i1

        x = torch.Tensor(xs[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
        y = torch.Tensor(ys[i1:i2]).to("cuda").to(torch.long)
        
        # random start
        delta = torch.rand_like(x, requires_grad=True).to("cuda")
        delta.data = delta.data * 2 * epsilon - epsilon
        delta.data = torch.clamp(x + delta.data, 0, 1) - x
        
        for t in range(num_iter):
            x_adv = x + delta
            
            layer_output = model.forward_until(x_adv, layer_i)
            layer_logits = model.linear_layers[layer_i](layer_output.reshape(layer_output.shape[0], -1))
            loss = nn.CrossEntropyLoss()(layer_logits, y)
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.data
                
                if grad_norm_type == 'sign':
                    grad_norm = grad.sign()
                elif grad_norm_type == 'l2':
                    grad_norm = grad / (torch.norm(grad, p=2, dim=(1,2,3), keepdim=True) + 1e-8)
                elif grad_norm_type == 'linf':
                    grad_norm = grad / (grad.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
                
                # Reset momentum buffer for each new batch size
                if momentum > 0:
                    if momentum_buffer is None or momentum_buffer.size(0) != current_batch_size:
                        momentum_buffer = grad_norm
                    else:
                        momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grad_norm
                    update = momentum_buffer
                else:
                    update = grad_norm

                delta.data = delta.data + alpha * update
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.data = torch.clamp(x + delta.data, 0, 1) - x
            
            # early stop
            with torch.no_grad():
                layer_output = model.forward_until(x + delta, layer_i)
                layer_logits = model.linear_layers[layer_i](layer_output.reshape(layer_output.shape[0], -1))
                pred = layer_logits.max(1)[1]
                if (pred != y).all():
                    break
            
            delta.grad.zero_()

        perturbed_image = torch.clamp(x + delta.data, 0, 1)
        all_perturbed_images.append(perturbed_image.cpu().numpy().transpose([0, 2, 3, 1]))
        
    return np.concatenate(all_perturbed_images, axis=0)


def fgsm_attack_layer_combined(model, xs, ys, epsilon, layer_idxs, layer_weights, batch_size=128):
    if layer_weights is None:
        layer_weights = [1.0 / len(layer_idxs)] * len(layer_idxs)  # equal weights
    layer_weights = np.array(layer_weights)
    layer_weights = layer_weights / np.sum(layer_weights)  # normalize to sum to 1

    all_perturbed_images = []
    its = int(np.ceil(xs.shape[0] / batch_size))

    for it in range(its):
        i1 = it * batch_size
        i2 = min([(it + 1) * batch_size, xs.shape[0]])

        x = torch.Tensor(xs[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
        y = torch.Tensor(ys[i1:i2]).to("cuda").to(torch.long)

        combined_grad = torch.zeros_like(x)
        for layer_idx, weight in zip(layer_idxs, layer_weights):
            x_copy = x.clone()
            x_copy.requires_grad = True

            layer_output = model.forward_until(x_copy, layer_idx)
            layer_logits = model.linear_layers[layer_idx](layer_output.reshape(layer_output.shape[0], -1))
            loss = nn.CrossEntropyLoss()(layer_logits, y)
            loss.backward()

            combined_grad += weight * x_copy.grad.data  # sum from all layers

        perturbed_image = x + epsilon * combined_grad.sign()
        perturbed_image = torch.clip(perturbed_image, 0, 1)

        all_perturbed_images.append(perturbed_image.detach().cpu().numpy().transpose([0, 2, 3, 1]))

    return np.concatenate(all_perturbed_images, axis=0)


def pgd_attack_layer_combined(model, xs, ys, epsilon, layer_idxs, layer_weights, alpha=0.01, num_iter=40, batch_size=128):
    if layer_weights is None:
        layer_weights = [1.0 / len(layer_idxs)] * len(layer_idxs)  # equal weights
    layer_weights = np.array(layer_weights)
    layer_weights = layer_weights / np.sum(layer_weights)  # normalize to sum to 1

    all_perturbed_images = []
    its = int(np.ceil(xs.shape[0] / batch_size))

    for it in range(its):
        i1 = it * batch_size
        i2 = min([(it + 1) * batch_size, xs.shape[0]])

        x = torch.Tensor(xs[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
        y = torch.Tensor(ys[i1:i2]).to("cuda").to(torch.long)

        delta = torch.zeros_like(x, requires_grad=True).to("cuda")
        delta.uniform_(-epsilon, epsilon)
        delta = torch.clamp(x + delta, 0, 1) - x

        for _ in range(num_iter):
            x_adv = x + delta
            combined_grad = torch.zeros_like(x)

            for layer_idx, weight in zip(layer_idxs, layer_weights):
                x_copy = x_adv.clone()
                x_copy.requires_grad = True

                layer_output = model.forward_until(x_copy, layer_idx)
                layer_logits = model.linear_layers[layer_idx](layer_output.reshape(layer_output.shape[0], -1))
                loss = nn.CrossEntropyLoss()(layer_logits, y)
                loss.backward()

                combined_grad += weight * x_copy.grad.data # sum from all layers

            delta = delta + alpha * combined_grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x
            
            delta = delta.detach()
            delta.requires_grad = True

        perturbed_image = torch.clamp(x + delta, 0, 1)
        all_perturbed_images.append(perturbed_image.detach().cpu().numpy().transpose([0, 2, 3, 1]))

    return np.concatenate(all_perturbed_images, axis=0)


def fgsm_attack_ensemble(model, images, labels, epsilon, batch_size):
    def get_cross_max_consensus_logits(outputs: torch.Tensor, k: int) -> torch.Tensor:
        Z_hat = outputs - outputs.max(dim=2, keepdim=True)[0]
        Z_hat = Z_hat - Z_hat.max(dim=1, keepdim=True)[0]
        Y, _ = torch.topk(Z_hat, k, dim=1)
        Y = Y[:, -1, :]
        return Y

    model = model.eval()
    model = model.cuda()
    perturbed_images = []

    for i in tqdm(range(0, len(images), batch_size), desc="fgsm ensemble", total=len(images) // batch_size, ncols=100):
        batch_images = images[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        x = torch.FloatTensor(batch_images.transpose(0, 3, 1, 2)).cuda()
        y = torch.LongTensor(batch_labels).cuda()
        x.requires_grad = True
        free_mem()

        layer_outputs = []
        for layer_i in layers_to_use:
            outputs = model.predict_from_layer(x, layer_i)
            layer_outputs.append(outputs.unsqueeze(1))
        ensemble_outputs = torch.cat(layer_outputs, dim=1)

        logits = get_cross_max_consensus_logits(ensemble_outputs, k=3)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        perturbed_image = x + epsilon * x.grad.data.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_images.append(perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1))

        x.grad.zero_()
    return np.concatenate(perturbed_images, axis=0)


def pgd_attack_ensemble(model, images, labels, epsilon, alpha=0.01, num_iter=40, batch_size=128):
    def get_cross_max_consensus_logits(outputs: torch.Tensor, k: int) -> torch.Tensor:
        Z_hat = outputs - outputs.max(dim=2, keepdim=True)[0]
        Z_hat = Z_hat - Z_hat.max(dim=1, keepdim=True)[0]
        Y, _ = torch.topk(Z_hat, k, dim=1)
        Y = Y[:, -1, :]
        return Y

    model = model.eval()
    model = model.cuda()
    perturbed_images = []

    for i in tqdm(range(0, len(images), batch_size), desc="pgd ensemble", total=len(images) // batch_size, ncols=100):
        batch_images = images[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        x = torch.FloatTensor(batch_images.transpose(0, 3, 1, 2)).cuda()
        y = torch.LongTensor(batch_labels).cuda()
        
        delta = torch.zeros_like(x, requires_grad=True).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = torch.clamp(x + delta, 0, 1) - x
        
        for _ in range(num_iter):
            x_adv = x + delta
            x_adv.requires_grad = True
            free_mem()

            layer_outputs = []
            for layer_i in layers_to_use:
                outputs = model.predict_from_layer(x_adv, layer_i)
                layer_outputs.append(outputs.unsqueeze(1))
            ensemble_outputs = torch.cat(layer_outputs, dim=1)

            logits = get_cross_max_consensus_logits(ensemble_outputs, k=3)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()

            grad = x_adv.grad.data
            delta = delta + alpha * grad.sign()
            
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x
            
            delta = delta.detach()
            delta.requires_grad = True

        perturbed_image = torch.clamp(x + delta, 0, 1)
        perturbed_images.append(perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1))

    return np.concatenate(perturbed_images, axis=0)


def hcaptcha_mask(images, mask: Image.Image, opacity: int):
    # opacity range: 0 (transparent) to 255 (opaque)
    def add_overlay(background: Image.Image, overlay: Image.Image, opacity: int) -> Image.Image:
        overlay = overlay.resize(background.size)
        result = Image.new("RGBA", background.size)
        result.paste(background, (0, 0))
        mask = Image.new("L", overlay.size, opacity)
        result.paste(overlay, (0, 0), mask)
        return result

    all_perturbed_images = []

    to_pil = lambda x: Image.fromarray((x * 255).astype(np.uint8))
    to_np = lambda x: np.array(x) / 255.0
    for i in tqdm(range(len(images)), desc="adding overlay", ncols=100):
        perturbed_image = to_np(add_overlay(to_pil(images[i]), mask, opacity).convert("RGB"))
        all_perturbed_images.append(perturbed_image)

    return np.array(all_perturbed_images)


def eval_layers(backbone_model, images_test_np, labels_test_np, layers_to_use):
    layer_acc = []
    for layer_i in layers_to_use:
        print(f"evaluating layer={layer_i}")
        linear_model = LinearNet(backbone_model, layer_i).to("cuda")
        linear_model.eval()
        test_hits, test_count, test_logits = eval_model(linear_model, images_test_np.copy(), labels_test_np.copy())
        layer_acc.append(test_hits / test_count)
    return {layer_i: acc for layer_i, acc in zip(layers_to_use, layer_acc)}


def eval_self_ensemble(backbone_model, images_test, labels_test, layers_to_use, batch_size=128):
    def get_cross_max_consensus_logits(outputs: torch.Tensor, k: int) -> torch.Tensor:
        Z_hat = outputs - outputs.max(dim=2, keepdim=True)[0]  # subtract the max per-predictor over classes
        Z_hat = Z_hat - Z_hat.max(dim=1, keepdim=True)[0]  # subtract the per-class max over predictors
        Y, _ = torch.topk(Z_hat, k, dim=1)  # get highest k values per class
        Y = Y[:, -1, :]  # get the k-th highest value per class
        assert Y.shape == (outputs.shape[0], outputs.shape[2])
        assert len(Y.shape) == 2
        return Y

    backbone_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images_test), batch_size), desc="ensemble eval", ncols=100):
            batch_images = torch.Tensor(images_test[i : i + batch_size].transpose([0, 3, 1, 2])).cuda()  # transpose to [batch_size, channels, height, width]
            batch_labels = labels_test[i : i + batch_size]
            layer_outputs = []

            for layer_i in layers_to_use:
                outputs = backbone_model.predict_from_layer(batch_images, layer_i)
                layer_outputs.append(outputs.unsqueeze(1))
            ensemble_outputs = torch.cat(layer_outputs, dim=1)  # [batch_size, num_layers, num_classes]

            logits = get_cross_max_consensus_logits(ensemble_outputs, k=3)
            predictions = torch.argmax(logits, dim=1)

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(batch_labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    return accuracy


def is_cached(filepath, combination):
    if filepath.exists() and filepath.stat().st_size > 0:
        lines = filepath.read_text().strip().split("\n")
        lines = [json.loads(line) for line in lines]
        for line in lines:
            if all(line[k] == v for k, v in combination.items()):
                return True
    return False


if __name__ == "__main__":
    fpath = output_path / "self_ensemble.jsonl"
    fpath.touch(exist_ok=True)

    combinations = {
        "dataset": ["cifar10", "cifar100", "imagenette"],
        # in future experiments: either set all to True or False, no combinations
        "training_noise": [False, True],
        "training_shuffle": [False, True],
        "training_adversarial": [False, True],
    }
    combs = list(product(*combinations.values()))
    for idx, comb in enumerate(combs):
        print(f"progress: {idx+1}/{len(combs)}")
        comb = {k: v for k, v in zip(combinations.keys(), comb)}
        if is_cached(fpath, comb):
            continue

        images_train_np, labels_train_np, images_test_np, labels_test_np, num_classes = get_dataset(comb["dataset"])

        resolutions = [32, 16, 8, 4]  # arbitrary resolutions to use in stacked images
        layers_to_use = [20, 30, 35, 40, 45, 50, 52]  # only some layers to save time -> anything below 20 is useless
        model = get_model(
            enable_noise=comb["training_noise"],
            enable_random_shuffle=comb["training_shuffle"],
            enable_adversarial_training=comb["training_adversarial"],
            resolutions=resolutions,
            layers_to_use=layers_to_use,
            num_classes=num_classes,
            images_train_np=images_train_np.copy(),
            labels_train_np=labels_train_np.copy(),
        )
        model.cuda()
        model.eval()
        free_mem()

        output = {
            **comb,
            # "plain_layer_accs": eval_layers(model, images_test_np.copy(), labels_test_np.copy(), layers_to_use),
            # "plain_ensemble_acc": eval_self_ensemble(model, images_test_np.copy(), labels_test_np.copy(), layers_to_use),
        }
        free_mem()

        # fgsm_idxs = [20, 30, 35, 40, 45, 50, 52]
        # for fgsm_idx in fgsm_idxs:
        #     fgsm_images_test_np = fgsm_attack_layer(model, images_test_np.copy()[:], labels_test_np.copy()[:], epsilon=8 / 255, layer_i=fgsm_idx, batch_size=64)
        #     output[f"fgsm_{fgsm_idx}_ensemble_acc"] = eval_self_ensemble(model, fgsm_images_test_np, labels_test_np.copy(), layers_to_use)
        #     output[f"fgsm_{fgsm_idx}_layer_accs"] = eval_layers(model, fgsm_images_test_np, labels_test_np.copy(), layers_to_use)
        # free_mem()

        # fgsmcombined_idxs = [20, 30, 35]
        # fgsmcombined_images_test_np = fgsm_attack_layer_combined(model, images_test_np.copy()[:], labels_test_np.copy()[:], epsilon=8 / 255, layer_idxs=fgsmcombined_idxs, layer_weights=None, batch_size=64)
        # output[f"fgsmcombined_{fgsmcombined_idxs}_ensemble_acc"] = eval_self_ensemble(model, fgsmcombined_images_test_np, labels_test_np.copy(), layers_to_use)
        # output[f"fgsmcombined_{fgsmcombined_idxs}_layer_accs"] = eval_layers(model, fgsmcombined_images_test_np, labels_test_np.copy(), layers_to_use)
        # free_mem()

        # fgsmensemble_images_test_np = fgsm_attack_ensemble(model, images_test_np.copy()[:], labels_test_np.copy()[:], epsilon=8 / 255, batch_size=64)
        # output["fgsmensemble_ensemble_acc"] = eval_self_ensemble(model, fgsmensemble_images_test_np, labels_test_np.copy(), layers_to_use)
        # output["fgsmensemble_layer_accs"] = eval_layers(model, fgsmensemble_images_test_np, labels_test_np.copy(), layers_to_use)
        # free_mem()

        pgd_idxs = [20, 30, 35, 40, 45, 50, 52]
        for pgd_idx in pgd_idxs:
            pgd_images_test_np = pgd_attack_layer(model, images_test_np.copy()[:], labels_test_np.copy()[:], epsilon=8 / 255, layer_i=pgd_idx, batch_size=64)
            output[f"pgd_{pgd_idx}_ensemble_acc"] = eval_self_ensemble(model, pgd_images_test_np, labels_test_np.copy(), layers_to_use)
            output[f"pgd_{pgd_idx}_layer_accs"] = eval_layers(model, pgd_images_test_np, labels_test_np.copy(), layers_to_use)
        free_mem()

        pgdcombined_idxs = [20, 30, 35]
        pgdcombined_images_test_np = pgd_attack_layer_combined(model, images_test_np.copy()[:], labels_test_np.copy()[:], epsilon=8 / 255, layer_idxs=pgdcombined_idxs, layer_weights=None, batch_size=64)
        output[f"pgdcombined_{pgdcombined_idxs}_ensemble_acc"] = eval_self_ensemble(model, pgdcombined_images_test_np, labels_test_np.copy(), layers_to_use)
        output[f"pgdcombined_{pgdcombined_idxs}_layer_accs"] = eval_layers(model, pgdcombined_images_test_np, labels_test_np.copy(), layers_to_use)

        pgdensemble_images_test_np = pgd_attack_ensemble(model, images_test_np.copy()[:], labels_test_np.copy()[:], epsilon=8 / 255, batch_size=64)
        output["pgdensemble_ensemble_acc"] = eval_self_ensemble(model, pgdensemble_images_test_np, labels_test_np.copy(), layers_to_use)
        output["pgdensemble_layer_accs"] = eval_layers(model, pgdensemble_images_test_np, labels_test_np.copy(), layers_to_use)
        free_mem()

        # opacities = [0, 1, 2, 4, 8, 16, 32, 64, 128, 255]
        # for opacity in opacities:
        #     hcaptcha_images_test_np = hcaptcha_mask(images_test_np.copy(), prerendered_mask, opacity)
        #     output[f"mask_{opacity}_layer_accs"] = eval_layers(model, hcaptcha_images_test_np, labels_test_np.copy(), layers_to_use)
        #     output[f"mask_{opacity}_ensemble_acc"] = eval_self_ensemble(model, hcaptcha_images_test_np, labels_test_np.copy(), layers_to_use)
        # free_mem()

        with fpath.open("a") as f:
            f.write(json.dumps(output) + "\n")
