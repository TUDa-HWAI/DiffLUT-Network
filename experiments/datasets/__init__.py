"""Unified public dataset interface for DiffLUT-Net experiments."""

from .loader import (
    DATASETS,
    SUPPORTED_DATASETS,
    input_dim_of_dataset,
    load_dataset,
    load_n,
    num_classes_of_dataset,
)

__all__ = [
    'DATASETS',
    'SUPPORTED_DATASETS',
    'input_dim_of_dataset',
    'load_dataset',
    'load_n',
    'num_classes_of_dataset',
]
