"""Small shared helpers: seeding, ranking, signs, top-k selection."""
from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and (if present) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch not installed / no cuda
        pass


def ranks_by_magnitude(attr: np.ndarray) -> np.ndarray:
    """Dense competition-free ordinal ranks by descending |attribution|.

    rank == 1 is the most important feature. Ties are broken stably by index so
    that two identical explanations always produce identical rank vectors.

    Returns an int array ``r`` with ``r[i]`` = rank of feature ``i``.
    """
    attr = np.asarray(attr, dtype=float).ravel()
    order = np.argsort(-np.abs(attr), kind="stable")
    ranks = np.empty(attr.shape[0], dtype=int)
    ranks[order] = np.arange(1, attr.shape[0] + 1)
    return ranks


def top_k_set(attr: np.ndarray, k: int) -> set[int]:
    """Indices of the top-``k`` features by |attribution|."""
    attr = np.asarray(attr, dtype=float).ravel()
    k = int(min(k, attr.shape[0]))
    order = np.argsort(-np.abs(attr), kind="stable")
    return set(order[:k].tolist())


def signs(attr: np.ndarray) -> np.ndarray:
    """Sign of each attribution; sign(0) == 0."""
    return np.sign(np.asarray(attr, dtype=float).ravel())
