r"""
Feature-index alignment
========================

The poster's key methodological control:

    "Mathematical alignment: all attributions projected onto the *same* feature
     indices, eliminating index-mismatch as a confound (Krishna et al., 2024)."

Different explainers can return attributions in different orders or
granularities. Before *any* metric is computed, every attribution vector must
live in one shared index space:

* **Tabular** (Adult Income): every method already returns one value per
  one-hot column, so alignment is the identity -- but we still pass everything
  through ``align_tabular`` to guarantee a fixed column order.

* **Image** (MNIST): perturbation methods (LIME, KernelSHAP) operate on coarse
  super-pixel patches, while gradient methods (IG, SmoothGrad) operate per
  pixel. ``align_to_superpixels`` pools the per-pixel gradient attributions into
  the same patch grid, so all four methods end up in one ``n_patches``-dim space.
"""
from __future__ import annotations

import numpy as np


def align_tabular(attr: np.ndarray, n_features: int) -> np.ndarray:
    """Identity alignment with a shape guarantee for tabular attributions."""
    attr = np.asarray(attr, dtype=float).ravel()
    if attr.shape[0] != n_features:
        raise ValueError(f"expected {n_features} features, got {attr.shape[0]}")
    return attr


def superpixel_grid(height: int, width: int, grid: int) -> np.ndarray:
    """Map each pixel to a patch id for a ``grid x grid`` super-pixel layout.

    Returns an int array of shape (height*width,) with values in
    ``[0, grid*grid)``. Patches are contiguous blocks; remainder rows/cols are
    folded into the last block so every pixel is assigned.
    """
    rows = np.minimum((np.arange(height) * grid) // height, grid - 1)
    cols = np.minimum((np.arange(width) * grid) // width, grid - 1)
    patch = rows[:, None] * grid + cols[None, :]
    return patch.ravel()


def align_to_superpixels(pixel_attr: np.ndarray, patch_of_pixel: np.ndarray,
                         n_patches: int, reduce: str = "mean") -> np.ndarray:
    """Pool per-pixel attributions into per-patch attributions.

    ``reduce='mean'`` averages the signed pixel attributions inside each patch
    (keeps direction, matching how super-pixel methods aggregate). ``'sum'`` is
    also available.
    """
    pixel_attr = np.asarray(pixel_attr, dtype=float).ravel()
    out = np.zeros(n_patches, dtype=float)
    counts = np.zeros(n_patches, dtype=float)
    np.add.at(out, patch_of_pixel, pixel_attr)
    np.add.at(counts, patch_of_pixel, 1.0)
    if reduce == "mean":
        counts[counts == 0] = 1.0
        out = out / counts
    return out
