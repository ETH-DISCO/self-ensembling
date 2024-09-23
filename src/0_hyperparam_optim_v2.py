import itertools
import json
from pathlib import Path

import torch
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import custom_torchvision
from dataloader import CIFAR10DataModule, CIFAR100DataModule
from utils import set_env

set_env(seed=41)

output_path = Path.cwd() / "data" / "hyperparams.jsonl"

batch_size = 32  # lower always better, but slower
train_ratio = 0.8  # common default

import pytorch_lightning as pl


class ResNetEnsemble(pl.LightningModule):
    def __init__(self, num_classes, lr, crossmax_k):
        super().__init__()
        self.net = custom_torchvision.resnet152_ensemble(num_classes=num_classes)
        custom_torchvision.set_resnet_weights(self.net, models.ResNet152_Weights.IMAGENET1K_V1)
        custom_torchvision.freeze_backbone(self.net)
        self.lr = lr
        self.crossmax_k = crossmax_k
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        losses = [self.criterion(outputs[:, i, :], labels) for i in range(len(self.net.fc_layers))]
        total_loss = sum(losses)
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = sum(self.criterion(outputs[:, i, :], labels) for i in range(len(self.net.fc_layers)))
        predictions = custom_torchvision.get_cross_max_consensus(outputs=outputs, k=self.crossmax_k)
        self.log("val_loss", loss)
        return {"val_loss": loss, "preds": predictions, "targets": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        accuracy = accuracy_score(targets.cpu(), preds.cpu())
        precision = precision_score(targets.cpu(), preds.cpu(), average="weighted")
        recall = recall_score(targets.cpu(), preds.cpu(), average="weighted")
        f1 = f1_score(targets.cpu(), preds.cpu(), average="weighted")
        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(config: dict):
    if config["dataset"] == "cifar10":
        datamodule = CIFAR10DataModule(batch_size, train_ratio)
    elif config["dataset"] == "cifar100":
        datamodule = CIFAR100DataModule(batch_size, train_ratio)

    model = ResNetEnsemble(num_classes=len(datamodule.classes), lr=config["lr"], crossmax_k=config["crossmax_k"])

    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"])],
        gpus=-1,
    )

    trainer.fit(model, datamodule=datamodule)

    results = {
        "config": config,
        "accuracy": trainer.callback_metrics["val_accuracy"].item(),
        "precision": trainer.callback_metrics["val_precision"].item(),
        "recall": trainer.callback_metrics["val_recall"].item(),
        "f1_score": trainer.callback_metrics["val_f1"].item(),
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
        "lr": [0.1],
        "num_epochs": [250],  # higher with early stopping is better, but slower (usually 200-300)
        "crossmax_k": [2],  # 2 because we assume vickery voting system (this can be tuned after training is done)
        "early_stopping_patience": [10],  # higher is better, but slower (usually 5-20)
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    print(f"searching {len(combinations)} combinations")

    for combination in combinations:

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
            print(f"skipping: {combination}")
            continue

        print(f"training: {combination}")
        train(config=combination)
