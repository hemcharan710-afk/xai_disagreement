"""
Generate the four poster attributions for an MNIST CNN, aligned to one grid.

The poster's alignment control in the image domain: perturbation methods (LIME,
KernelSHAP) act on coarse **super-pixel patches**, while gradient methods (IG,
SmoothGrad) act per pixel. We project *everything* onto the same ``grid x grid``
super-pixel space (default 7x7 = 49 patches):

    * LIME / KernelSHAP -- native 49-dim: a patch is "present" (original pixels)
      or "absent" (black baseline). The perturbation predict-fn rebuilds the
      image from a patch mask.
    * IG / SmoothGrad   -- pixel attributions pooled (mean) into the 49 patches.

So all four methods produce a 49-dim signed attribution and disagreement is
measured in one shared index space.
"""
from __future__ import annotations

import numpy as np

from xai_disagreement import explainers as E, alignment as A


def _patch_predict_factory(image: np.ndarray, patch_of_pixel: np.ndarray,
                           n_patches: int, model):
    """Closure: patch-mask matrix (n, n_patches) -> model class probabilities."""
    baseline = np.zeros_like(image)  # black = absent (natural for MNIST)

    def predict(Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        pix = Z[:, patch_of_pixel]                       # (n, 784) keep-weights
        imgs = image[None, :] * pix + baseline[None, :] * (1 - pix)
        return model.predict_proba(imgs.astype(np.float32))
    return predict


def explain_model(trained, ds, instances: np.ndarray, grid: int = 7,
                  n_perturb: int = 600, ig_steps: int = 32, sg_samples: int = 20,
                  seed: int = 0) -> tuple[dict[str, np.ndarray], int]:
    """Return (``method -> (n_instances, n_patches)`` attributions, n_patches)."""
    model = trained.model
    rng = np.random.default_rng(seed)
    h, w = ds.img_hw
    patch_of_pixel = A.superpixel_grid(h, w, grid)
    n_patches = grid * grid
    preds = model.predict_proba(instances).argmax(1)

    out = {m: np.zeros((len(instances), n_patches)) for m in E.POSTER_METHODS}
    ones = np.ones(n_patches)
    bg = np.zeros((64, n_patches))  # "absent" background for perturbation methods

    for i, x in enumerate(instances):
        t = int(preds[i])
        pp = _patch_predict_factory(x, patch_of_pixel, n_patches, model)
        out["LIME"][i] = E.lime(ones, pp, t, bg, n_samples=n_perturb, rng=rng)
        out["KernelSHAP"][i] = E.kernel_shap(ones, pp, t, bg,
                                             n_samples=n_perturb, rng=rng)
        ig_pix = E.integrated_gradients(model, x, t, steps=ig_steps)
        # SmoothGrad * Input keeps the two gradient methods on a shared x*grad
        # basis so they cluster as one family (poster Finding 3).
        sg_pix = E.smoothgrad(model, x, t, n_samples=sg_samples, rng=rng,
                              times_input=True)
        out["IG"][i] = A.align_to_superpixels(ig_pix, patch_of_pixel, n_patches)
        out["SmoothGrad"][i] = A.align_to_superpixels(sg_pix, patch_of_pixel,
                                                      n_patches)
    return out, n_patches
