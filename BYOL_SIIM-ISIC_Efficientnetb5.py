# data augmentations
import random
from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf
import torch
from torch import nn, Tensor
import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Union, Callable, Tuple
from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as f
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
import torch
import torchvision
import torchvision.transforms as transforms
from os import cpu_count

from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.models import resnet18, resnet101
from numpy import random


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


seed = 42
seed_torch(seed)


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def default_augmentation(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    return nn.Sequential(
        tf.Resize(size=image_size),
        RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        aug.RandomGrayscale(p=0.2),
        aug.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        aug.RandomResizedCrop(size=image_size),
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )


def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    def _hook(self, _, __, output):
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            self._projector_dim = output.shape[-1]
        self._encoded = self.projector(output)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) -> Tensor:
        _ = self.model(x)
        return self._encoded


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = f.normalize(x, dim=-1)
    y = f.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (128, 128),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.999,
        **hparams,
    ):
        super().__init__()
        self.augment = (
            default_augmentation(image_size) if augment_fn is None else augment_fn
        )
        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        self.hparams.update(hparams)
        self._target = None

        self.encoder(torch.zeros(2, 3, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(self.encoder(x))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    # --- Methods required for PyTorch Lightning only! ---

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        pred1, pred2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.target(x1), self.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("train_loss", loss.item())
        self.update_target()

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        x1, x2 = self.augment(x), self.augment(x)
        pred1, pred2 = self.forward(x1), self.forward(x2)
        targ1, targ2 = self.target(x1), self.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())


class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, **hparams):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        loss = f.cross_entropy(self.forward(x), y)
        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        loss = f.cross_entropy(self.forward(x), y)
        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())


class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        if self.mode == "test":
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


def accuracy(pred: Tensor, labels: Tensor) -> float:
    return (pred.argmax(dim=-1) == labels).float().mean().item()


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
batch_size = 25

data_folder = "512"
data_dir = "/blue/daisyw/iharmon1/data/SIIM-ISIC/data/"
savepath = "."
# 2020 data
df_train = pd.read_csv(
    os.path.join(data_dir, f"jpeg-melanoma-{data_folder}x{data_folder}", "train.csv")
)
df_train = df_train[df_train["tfrecord"] != -1].reset_index(drop=True)
df_train["filepath"] = df_train["image_name"].apply(
    lambda x: os.path.join(
        data_dir, f"jpeg-melanoma-{data_folder}x{data_folder}/train", f"{x}.jpg"
    )
)

# 2018, 2019 data (external data)
df_train2 = pd.read_csv(
    os.path.join(data_dir, f"jpeg-isic2019-{data_folder}x{data_folder}", "train.csv")
)
df_train2 = df_train2[df_train2["tfrecord"] >= 0].reset_index(drop=True)
df_train2["filepath"] = df_train2["image_name"].apply(
    lambda x: os.path.join(
        data_dir, f"jpeg-isic2019-{data_folder}x{data_folder}/train", f"{x}.jpg"
    )
)

# concat train data
df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)
df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)

print(df_train)

TRAIN_DATASET = MelanomaDataset(df_train, mode="train", meta_features=None)
total = len(df_train)

train_ratio = 0.80

train_set, val_set = random_split(
    TRAIN_DATASET,
    [int(total * train_ratio), total - int(total * train_ratio)],
    generator=torch.Generator().manual_seed(seed),
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=5,
    pin_memory=True,
)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5, pin_memory=True)

##Supervision
model = EfficientNet.from_pretrained("efficientnet-b5")

supervised = SupervisedLightningModule(model)
trainer = pl.Trainer(max_epochs=10, gpus=1, profiler="simple")
trainer.fit(supervised, train_loader, val_loader)

torch.save(model.state_dict(), savepath + "/EfficientnetB5")

model.cuda()
acc = sum([accuracy(model(x.cuda()), y.cuda()) for x, y in val_loader]) / len(
    val_loader
)
print(f"Accuracy without BYOL self-supervision: {acc:.3f}")

del model


##Self-Supervision

model = EfficientNet.from_pretrained("efficientnet-b5")
byol = BYOL(model, image_size=(int(data_folder) - 64, int(data_folder) - 64))
trainer = pl.Trainer(
    max_epochs=10, gpus=1, accumulate_grad_batches=2048 // 128, profiler="simple"
)
trainer.fit(byol, train_loader, val_loader)

torch.save(model.state_dict(), savepath + "/BYOL_Selfsupervision_EfficientnetB5")
del model
##Supervision


model = EfficientNet.from_pretrained("efficientnet-b5")
model.load_state_dict(torch.load(savepath + "/BYOL_Selfsupervision_EfficientnetB5"))

supervised = SupervisedLightningModule(model)
trainer = pl.Trainer(max_epochs=10, gpus=1, profiler="simple")
trainer.fit(supervised, train_loader, val_loader)


model.cuda()
acc = sum([accuracy(model(x.cuda()), y.cuda()) for x, y in val_loader]) / len(
    val_loader
)
print(f"Accuracy with BYOL self-supervision: {acc:.3f}")

torch.save(model.state_dict(), savepath + "/BYOL_EfficientnetB5")

