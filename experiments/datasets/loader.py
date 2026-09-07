"""Single entry point for all supported experiment datasets."""

import importlib
import math
from pathlib import Path

import numpy as np
import torch
import torchvision

from .array_dataset import ArrayDataset


DATA_ROOT = Path(__file__).resolve().parent

# This registry is the public dataset API. Keep CLI choices and metadata derived
# from it so a dataset cannot be added in one place and forgotten in another.
DATASETS = {
    "cifar-10-8-thresholds": {
        "kind": "cifar10", "augmented": False, "input_dim": 3 * 32 * 32 * 8,
    },
    "cifar-10-8-thresholds-aug": {
        "kind": "cifar10", "augmented": True, "input_dim": 3 * 32 * 32 * 8,
    },
    "fashion_mnist-8thresholds": {
        "kind": "fashion_mnist", "augmented": False, "input_dim": 28 * 28 * 8,
    },
    "mnist-8thresholds": {
        "kind": "mnist", "augmented": False, "input_dim": 28 * 28 * 8,
    },
    "mnist-aug-8thresholds": {
        "kind": "mnist", "augmented": True, "input_dim": 28 * 28 * 8,
    },
    "jsc-openml": {
        "kind": "array",
        "directory": "jsc-openml",
        "downloader": "experiments.datasets.download_jsc_openml",
        "input_dim": 16 * 200,
    },
    "jsc-cernbox": {
        "kind": "array",
        "directory": "jsc-cernbox",
        "downloader": "experiments.datasets.download_jsc_cernbox",
        "input_dim": 16 * 500,
    },
}

SUPPORTED_DATASETS = tuple(DATASETS)


class DistributiveThermometer:
    """Feature-wise quantile thermometer encoding for image tensors."""

    def __init__(self, num_bits=8):
        self.num_bits = int(num_bits)
        self.thresholds = None

    def fit(self, x):
        indices = [
            int(x.shape[0] * i / (self.num_bits + 1))
            for i in range(1, self.num_bits + 1)
        ]
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        # Unlike torch.sort, numpy.partition does not allocate a full int64
        # index tensor, which is especially expensive for CIFAR-10.
        partitioned = np.partition(x, kth=indices, axis=0)
        self.thresholds = torch.from_numpy(partitioned[indices])
        return self

    def binarize(self, x):
        if self.thresholds is None:
            raise RuntimeError("DistributiveThermometer must be fitted first")
        binary = (x.unsqueeze(0) > self.thresholds).float()
        return binary.flatten(0, 1)


def _dataset_spec(name):
    try:
        return DATASETS[name]
    except KeyError:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset {name!r}. Choose one of: {supported}")


def _image_dataset_class(kind):
    return {
        "mnist": torchvision.datasets.MNIST,
        "fashion_mnist": torchvision.datasets.FashionMNIST,
        "cifar10": torchvision.datasets.CIFAR10,
    }[kind]


def _image_root(kind):
    return DATA_ROOT / {
        "mnist": "data-mnist",
        "fashion_mnist": "data-fashion-mnist",
        "cifar10": "data-cifar",
    }[kind]


def _fit_image_thermometer(dataset, kind):
    data = dataset.data
    print(f"[*] Fitting 8-threshold thermometer for {kind}...")
    thermometer = DistributiveThermometer(num_bits=8).fit(data)
    if kind == "cifar10":
        # CIFAR is stored as NHWC and its final transform normalizes to [-1, 1].
        thermometer.thresholds = (
            thermometer.thresholds.permute(0, 3, 1, 2)
            .float().div_(255).mul_(2).sub_(1)
        )
    else:
        thermometer.thresholds = thermometer.thresholds.unsqueeze(1).float().div_(255)
    print(
        "[*] Threshold range: "
        f"{thermometer.thresholds.min().item():.4f} to "
        f"{thermometer.thresholds.max().item():.4f}"
    )
    return thermometer


def _image_transforms(kind, augmented, thermometer):
    transforms = torchvision.transforms
    train_steps = []

    if augmented and kind == "cifar10":
        train_steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    elif augmented:
        train_steps.append(
            transforms.RandomAffine(
                degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=10,
            )
        )

    common_steps = [transforms.ToTensor()]
    if kind == "cifar10":
        common_steps.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    common_steps.append(transforms.Lambda(thermometer.binarize))

    train_transform = transforms.Compose(train_steps + common_steps)
    evaluation_transform = transforms.Compose(common_steps)
    return train_transform, evaluation_transform


def _load_image_sets(spec):
    kind = spec["kind"]
    dataset_class = _image_dataset_class(kind)
    root = str(_image_root(kind))

    raw_train_set = dataset_class(root, train=True, download=True)
    thermometer = _fit_image_thermometer(raw_train_set, kind)
    train_transform, evaluation_transform = _image_transforms(
        kind, spec["augmented"], thermometer,
    )

    train_set = dataset_class(root, train=True, transform=train_transform)
    validation_base = dataset_class(root, train=True, transform=evaluation_transform)
    test_set = dataset_class(
        root, train=False, download=True, transform=evaluation_transform,
    )
    return train_set, validation_base, test_set


def _load_array_sets(spec):
    data_dir = DATA_ROOT / spec["directory"]
    paths = {
        "train_data": data_dir / "train_data.npy",
        "train_labels": data_dir / "train_labels.npy",
        "test_data": data_dir / "test_data.npy",
        "test_labels": data_dir / "test_labels.npy",
    }

    if not all(path.exists() for path in paths.values()):
        downloader = importlib.import_module(spec["downloader"])
        downloader.download_and_process()

    train_set = ArrayDataset(paths["train_data"], paths["train_labels"])
    test_set = ArrayDataset(paths["test_data"], paths["test_labels"])
    return train_set, train_set, test_set


def _split_train_validation(train_set, validation_base, valid_fraction, seed):
    if not 0 <= valid_fraction < 1:
        raise ValueError("valid_set_size must be in the range [0, 1)")

    train_size = math.ceil((1 - valid_fraction) * len(train_set))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_set), generator=generator).tolist()
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:]
    return (
        torch.utils.data.Subset(train_set, train_indices),
        torch.utils.data.Subset(validation_base, validation_indices),
    )


def _make_loaders(args, train_set, validation_set, test_set):
    common = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "num_workers": getattr(args, "num_workers", 4),
    }
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, drop_last=True, **common,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, shuffle=False, drop_last=False, **common,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, drop_last=False, **common,
    )
    return train_loader, validation_loader, test_loader


def load_dataset(args):
    """Load one registered dataset and return train/validation/test loaders."""
    spec = _dataset_spec(args.dataset)
    if spec["kind"] == "array":
        train_set, validation_base, test_set = _load_array_sets(spec)
    else:
        train_set, validation_base, test_set = _load_image_sets(spec)

    train_set, validation_set = _split_train_validation(
        train_set,
        validation_base,
        getattr(args, "valid_set_size", 0.0),
        getattr(args, "seed", 0),
    )
    return _make_loaders(args, train_set, validation_set, test_set)


def load_n(loader, n):
    """Cycle over a loader until exactly ``n`` batches have been yielded."""
    if len(loader) == 0:
        raise ValueError("Cannot load batches from an empty DataLoader")
    yielded = 0
    while yielded < n:
        for batch in loader:
            yield batch
            yielded += 1
            if yielded == n:
                return


def input_dim_of_dataset(dataset):
    spec = _dataset_spec(dataset)
    if spec["kind"] == "array":
        data_dir = DATA_ROOT / spec["directory"]
        try:
            return ArrayDataset.get_input_dim(str(data_dir))
        except (OSError, KeyError, ValueError):
            pass
    return spec["input_dim"]


def num_classes_of_dataset(dataset):
    spec = _dataset_spec(dataset)
    return 5 if spec["kind"] == "array" else 10
