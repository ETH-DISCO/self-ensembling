import copy
import hashlib
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, models
from torchvision.models import resnet152
from tqdm import tqdm

assert torch.cuda.is_available()


#
# config
#


classes_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(classes_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


#
# seed
#


seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


@contextmanager
def isolated_environment():
    # save and restore random states in a context manager
    # used to separate random-seed-fixing behavior from the attacks later
    np_random_state = np.random.get_state()
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_print_options = np.get_printoptions()
    try:
        yield  # execute code block
    finally:
        np.random.set_state(np_random_state)
        random.setstate(python_random_state)
        torch.set_rng_state(torch_random_state)
        if cuda_random_state:
            torch.cuda.set_rng_state_all(cuda_random_state)
        np.set_printoptions(**numpy_print_options)


#
# data
#


classes = 100

if classes == 10:
    trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    testset = datasets.CIFAR10(root=dataset_path, train=False, download=True)
    original_images_train_np = np.array(trainset.data)
    original_labels_train_np = np.array(trainset.targets)
    original_images_test_np = np.array(testset.data)
    original_labels_test_np = np.array(testset.targets)
elif classes == 100:
    trainset = datasets.CIFAR100(root=dataset_path, train=True, download=True)
    testset = datasets.CIFAR100(root=dataset_path, train=False, download=True)
    original_images_train_np = np.array(trainset.data)
    original_labels_train_np = np.array(trainset.targets)
    original_images_test_np = np.array(testset.data)
    original_labels_test_np = np.array(testset.targets)
else:
    assert False

images_train_np = original_images_train_np / 255.0  # scale to [0, 1]
images_test_np = original_images_test_np / 255.0
labels_train_np = original_labels_train_np
labels_test_np = original_labels_test_np


#
# preprocessing
#


def custom_rand(input_tensor, size):
    return torch.Tensor(np.random.rand(*size)).to("cuda")


def custom_choices(items, tensor):
    return np.random.choice(items, (len(tensor)))


resolutions = [32, 16, 8, 4]  # pretty arbitrary
down_noise = 0.2  # noise standard deviation to be added at the low resolution
up_noise = 0.2  # noise stadard deviation to be added at the high resolution
jit_size = 3  # max size of the x-y jit in each axis, sampled uniformly from -jit_size to +jit_size inclusive
shuffle_image_versions_randomly = False  # random shuffling of multi-res images (false in paper but good for experiments)


def default_make_multichannel_input(images):  # channel wise image stack: just repeat the same image for each resolution (unused)
    return torch.concatenate([images] * len(resolutions), axis=1)


def make_multichannel_input(images):  # channel wise image stack + natural noise
    all_channels = []

    for i, r in enumerate(resolutions):
        down_res = r
        jits_x = custom_choices(range(-jit_size, jit_size + 1), images + i)  # x-shift
        jits_y = custom_choices(range(-jit_size, jit_size + 1), 51 * images + 7 * i + 125 * r)  # y-shift
        contrasts = custom_choices(np.linspace(0.7, 1.5, 100), 7 + 3 * images + 9 * i + 5 * r)  # change in contrast
        color_amounts = contrasts = custom_choices(np.linspace(0.5, 1.0, 100), 5 + 7 * images + 8 * i + 2 * r)  # change in color amount

        def apply_transformations(images, down_res=224, up_res=224, jit_x=0, jit_y=0, down_noise=0.0, up_noise=0.0, contrast=1.0, color_amount=1.0):
            images_collected = []
            for i in range(images.shape[0]):
                image = images[i]
                image = torchvision.transforms.functional.adjust_contrast(image, contrast[i])  # contrast
                image = torch.roll(image, shifts=(jit_x[i], jit_y[i]), dims=(-2, -1))  # x, y jitter
                image = color_amount[i] * image + torch.mean(image, axis=0, keepdims=True) * (1 - color_amount[i])  # grayscaling
                images_collected.append(image)
            images = torch.stack(images_collected, axis=0)

            images = F.interpolate(images, size=(down_res, down_res), mode="bicubic")  # downscale
            noise = down_noise * custom_rand(images + 312, (images.shape[0], 3, down_res, down_res)).to("cuda")  # low res noise
            images = images + noise
            images = F.interpolate(images, size=(up_res, up_res), mode="bicubic")  # upscale
            noise = up_noise * custom_rand(images + 812, (images.shape[0], 3, up_res, up_res)).to("cuda")  # high res noise
            images = images + noise
            images = torch.clip(images, 0, 1)  # clip to right range
            return images

        images_now = apply_transformations(images, down_res=down_res, up_res=32, jit_x=jits_x, jit_y=jits_y, down_noise=down_noise, up_noise=up_noise, contrast=contrasts, color_amount=color_amounts)
        all_channels.append(images_now)

    if not shuffle_image_versions_randomly:
        return torch.concatenate(all_channels, axis=1)
    elif shuffle_image_versions_randomly:
        indices = torch.randperm(len(all_channels))
        shuffled_tensor_list = [all_channels[i] for i in indices]
        return torch.concatenate(shuffled_tensor_list, axis=1)


#
# vanilla resnet: train batch eval
#


def eval_model(model, images_in, labels_in, batch_size=128):
    all_preds = []
    all_logits = []

    with torch.no_grad():
        its = int(np.ceil(float(len(images_in)) / float(batch_size)))  # iterations
        pbar = tqdm(range(its), desc="evaluation", ncols=100)  # progress bar
        for it in pbar:
            i1 = it * batch_size
            i2 = min([(it + 1) * batch_size, len(images_in)])

        inputs = torch.Tensor(images_in[i1:i2].transpose([0, 3, 1, 2])).to("cuda")
        outputs = model(inputs)

        all_logits.append(outputs.detach().cpu().numpy())
        preds = torch.argmax(outputs, axis=-1)  # get the index of the max logit from self ensemble
        all_preds.append(preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    return np.sum(all_preds == labels_in), all_preds.shape[0], all_logits


#
# vanilla resnet: tuning
#


def fgsm_attack(model, xs, ys, epsilon, random_reps=1, batch_size=64):  # for light adv training (unused)
    model = model.eval()

    its = int(np.ceil(xs.shape[0] / batch_size))

    all_perturbed_images = []

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


def train_model(
    model_in,
    images_in,
    labels_in,
    epochs=10,
    lr=1e-3,
    batch_size=512,
    optimizer_in=optim.Adam,
    subset_only=None,
    mode="eval",
    use_adversarial_training=False,
    adversarial_epsilon=8 / 255,
    skip_test_set_eval=False,
):
    global storing_models

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
        randomized_ids = np.random.permutation(range(len(images_in)))

        if mode == "train": # sometimes switches due to black-box evals
            model_in.train()
        elif mode == "eval":
            model_in.eval()
        else:
            assert False

        its = int(np.ceil(float(len(images_in)) / float(batch_size)))
        pbar = tqdm(range(its), desc="Training", ncols=100)

        all_hits = []

        for it in pbar:
            i1 = it * batch_size
            i2 = min([(it + 1) * batch_size, len(images_in)])

            ids_now = randomized_ids[i1:i2]

            np_images_used = images_in[ids_now]
            np_labels_used = labels_in[ids_now]

            inputs = torch.Tensor(np_images_used.transpose([0, 3, 1, 2])).to("cuda")

            if use_adversarial_training:
                attacked_images = fgsm_attack(
                    model_in.eval(),
                    np_images_used[:],
                    np_labels_used[:],
                    epsilon=adversarial_epsilon,
                    random_reps=1,
                    batch_size=batch_size // 2,
                )
                np_images_used = attacked_images
                np_labels_used = np_labels_used

                if mode == "train":
                    model_in.train()
                elif mode == "eval":
                    model_in.eval()

            inputs = torch.Tensor(np_images_used.transpose([0, 3, 1, 2])).to("cuda")
            labels = torch.Tensor(np_labels_used).to("cuda").to(torch.long)

            # zero the parameter gradients
            train_optimizer.zero_grad()

            inputs_used = inputs

            # the actual optimization step
            outputs = model_in(inputs_used)
            loss = criterion(outputs, labels)
            loss.backward()
            train_optimizer.step()

            # train set batch stats
            preds = torch.argmax(outputs, axis=-1)
            acc = torch.mean((preds == labels).to(torch.float), axis=-1)
            all_hits.append((preds == labels).to(torch.float).detach().cpu().numpy())
            train_accs.append(acc.detach().cpu().numpy())
            pbar.set_description(f"train acc={acc.detach().cpu().numpy()} loss={loss.item()}")

        if not skip_test_set_eval:
            with isolated_environment():
                eval_model_copy = copy.deepcopy(model_in)
                test_hits, test_count, _ = eval_model(eval_model_copy.eval(), images_test_np, labels_test_np)
        else:
            # to avoid dividing by zero
            test_hits = 0
            test_count = 1

        # end of epoch eval
        train_hits = np.sum(np.concatenate(all_hits, axis=0).reshape([-1]))
        train_count = np.concatenate(all_hits, axis=0).reshape([-1]).shape[0]
        print(f"e={epoch} train {train_hits} / {train_count} = {train_hits/train_count},  test {test_hits} / {test_count} = {test_hits/test_count}")

        test_accs.append(test_hits / test_count)

    print("\nFinished Training")

    return model_in


state_dict = models.ResNet152_Weights.IMAGENET1K_V1.get_state_dict(progress=True, model_dir=weights_path)
imported_model = resnet152(weights=state_dict)

# fix first conv layer
in_planes = 3
planes = 64
stride = 2
N = len(resolutions)  # input channels multiplier due to multi-res input
conv2 = nn.Conv2d(N * in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
imported_model.conv1 = copy.deepcopy(conv2)  # replace first conv layer with multi-res one


class ImportedModelWrapper(nn.Module):
    def __init__(self, imported_model, multichannel_fn):
        super(ImportedModelWrapper, self).__init__()
        self.imported_model = imported_model
        self.multichannel_fn = multichannel_fn

    def forward(self, x):
        # preprocss
        x = self.multichannel_fn(x)
        x = F.interpolate(x, size=(224, 224), mode="bicubic")
        x = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406] * (x.shape[1] // 3), std=[0.229, 0.224, 0.225] * (x.shape[1] // 3))(x)
        # forward
        x = self.imported_model(x)
        return x


wrapped_model = ImportedModelWrapper(imported_model, make_multichannel_input).to("cuda")
wrapped_model.multichannel_fn = make_multichannel_input  # huh?

lr = 3.3e-5  # found with very simple "grid search" by hand, likely not optimal!
mode = "train"

epochs = 6

model = copy.deepcopy(wrapped_model)
model.multichannel_fn = make_multichannel_input

if mode == "eval":
    model = model.eval()
elif mode == "train":
    model = model.train()
else:
    assert False

train_accs = []
test_accs = []

torch.cuda.empty_cache()

device = torch.device("cuda:0")

# with torch.autocast("cuda"):
model = train_model(
    model,
    images_train_np,
    labels_train_np,
    epochs=epochs,
    lr=lr,
    optimizer_in=optim.Adam,
    batch_size=128,
    mode=mode,
)


# 
# self ensembling
# 


class BatchNormLinear(nn.Module): # faster training by normalizing layer inputs through scaling
    def __init__(self, in_features, out_features, device="cuda"):
        super(BatchNormLinear, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_features, device=device)
        self.linear = nn.Linear(in_features, out_features, device=device)

    def forward(self, x):
        x = self.batch_norm(x)
        return self.linear(x)


class WrapModelForResNet152(torch.nn.Module):
    def __init__(self, model, multichannel_fn, classes=10):
        super(WrapModelForResNet152, self).__init__()

        self.multichannel_fn = multichannel_fn
        self.model = model
        self.classes = classes

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
        self.linear_layers = torch.nn.ModuleList([BatchNormLinear(self.all_dims[i], classes, device="cuda") for i in range(len(self.all_dims))])

    def prepare_input(self, x):
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

    def predict_from_layer(self, x, l): # entry a) for self-ensembling
        x = self.forward_until(x, l)
        x = x.reshape([x.shape[0], -1])
        return self.linear_layers[l](x)

    def predict_from_several_layers(self, x, layers): # entry b) for self-ensembling
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


resnet152_wrapper = WrapModelForResNet152(model.imported_model, make_multichannel_input, classes=classes)
resnet152_wrapper.multichannel_fn = make_multichannel_input
# del model # for memory reasons
resnet152_wrapper = resnet152_wrapper.to("cuda")

for layer_i in range(53):
    print(f"layer={layer_i} {resnet152_wrapper.predict_from_layer(torch.Tensor(np.zeros((2,3,32,32))).cuda(),layer_i).shape}")

class LinearNet(nn.Module): # single intermediate output
    def __init__(self, model, layer_i):
        super(LinearNet, self).__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, inputs):
        return self.model.predict_from_layer(inputs, self.layer_i)

backbone_model = copy.deepcopy(resnet152_wrapper)
del resnet152_wrapper
# only training some layers to save time -- super early ones are bad on anything harder than CIFAR-10
layers_to_use = [20, 30, 35, 40, 45, 50, 52]
lr = 3.3e-5  # random stuff again
epochs = 1
batch_size = 64  # for CUDA RAM reasons

mode = "train"
backbone_model.eval()
linear_model = LinearNet(backbone_model, 5).to("cuda")  # just to have it ready
torch.cuda.empty_cache()

device = torch.device("cuda:0")

linear_layers_collected_dict = dict()

for layer_i in reversed(layers_to_use):
    print(f"///////// layer={layer_i} ///////////")

    linear_model.layer_i = layer_i
    linear_model.fixed_mode = "train"

    train_accs = []
    test_accs = []
    robust_accs = []
    clean_accs = []
    actual_robust_accs = []

    all_models = []

    torch.cuda.empty_cache()

    linear_model = train_model(
        linear_model,
        images_train_np[:],
        labels_train_np[:],
        epochs=epochs,
        lr=lr,
        optimizer_in=optim.Adam,
        batch_size=batch_size,
        mode=mode,
        subset_only=linear_model.model.linear_layers[layer_i].parameters(),  # just the linear projection
        use_adversarial_training=False,
        adversarial_epsilon=None,
    )

    linear_layers_collected_dict[layer_i] = copy.deepcopy(backbone_model.linear_layers[layer_i])
