"""
MNIST loader for the image-domain extension.

Images are kept as flat 784-vectors in [0, 1] (the CNN reshapes internally), so
the same flat-vector explainers used for the tabular study apply unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Dataset:
    X_train: np.ndarray   # (n, 784) float32 in [0,1]
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int = 10
    n_features: int = 784
    img_hw: tuple[int, int] = (28, 28)


def load_mnist(n_train: int = 6000, n_test: int = 1000, seed: int = 0
               ) -> Dataset:
    """Load (and cache) MNIST via torchvision, subsampled for speed."""
    from torchvision import datasets

    os.makedirs(CACHE, exist_ok=True)
    tr = datasets.MNIST(CACHE, train=True, download=True)
    te = datasets.MNIST(CACHE, train=False, download=True)

    Xtr = (tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0)
    ytr = tr.targets.numpy().astype(int)
    Xte = (te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0)
    yte = te.targets.numpy().astype(int)

    rng = np.random.default_rng(seed)
    itr = rng.choice(len(Xtr), size=min(n_train, len(Xtr)), replace=False)
    ite = rng.choice(len(Xte), size=min(n_test, len(Xte)), replace=False)
    return Dataset(Xtr[itr], Xte[ite], ytr[itr], yte[ite])
