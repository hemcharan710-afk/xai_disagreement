r"""
Explanation methods
===================

The four methods our poster applies uniformly, plus the two extra gradient
methods used in the Krishna et al. reproduction. Everything returns a 1-D numpy
array of **signed** attributions for a single instance and target class.

Two families (this split is exactly Finding 3 of the poster):

    Perturbation family   LIME, KernelSHAP
        model accessed only through a ``predict_proba(X) -> (n, C)`` callable,
        so they work for *any* model (sklearn or torch).

    Gradient family       VanillaGradient, Gradient*Input,
                          IntegratedGradients, SmoothGrad
        require a differentiable ``torch.nn.Module``.

LIME and KernelSHAP are implemented from scratch (weighted local linear
regression) to keep the repo dependency-light and to make the math explicit;
they coincide with the ``lime`` / ``shap`` libraries up to sampling noise.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


PredictProba = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# Perturbation family
# --------------------------------------------------------------------------- #
def lime(x: np.ndarray, predict_proba: PredictProba, target: int,
         background: np.ndarray, n_samples: int = 1000,
         kernel_width: float | None = None, rng: np.random.Generator | None = None
         ) -> np.ndarray:
    r"""LIME for tabular / flattened data (local linear approximation).

    Perturbations toggle each feature between its true value (mask=1) and a value
    drawn from ``background`` (mask=0). The local surrogate is a linear model fit
    on the masks, weighted by an exponential proximity kernel in mask space. The
    surrogate coefficients are the attributions.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float).ravel()
    d = x.shape[0]
    kernel_width = kernel_width or np.sqrt(d) * 0.75

    masks = rng.integers(0, 2, size=(n_samples, d)).astype(float)
    masks[0] = 1.0  # always include the instance itself
    bg_idx = rng.integers(0, background.shape[0], size=(n_samples, d))
    bg_vals = background[bg_idx, np.arange(d)[None, :]]
    Z = np.where(masks > 0, x[None, :], bg_vals)

    y = predict_proba(Z)[:, target]
    dist = np.sqrt(np.sum((1.0 - masks) ** 2, axis=1))
    weights = np.exp(-(dist ** 2) / (kernel_width ** 2))

    return _weighted_linear_coefs(masks, y, weights)


def kernel_shap(x: np.ndarray, predict_proba: PredictProba, target: int,
                background: np.ndarray, n_samples: int = 1000,
                rng: np.random.Generator | None = None) -> np.ndarray:
    r"""KernelSHAP: Shapley values via the SHAP-kernel weighted linear regression.

    Coalitions ``z in {0,1}^d`` decide which features take their true value vs a
    background value. The surrogate is fit with the SHAP kernel weight
    ``(d-1) / (C(d,|z|) * |z| * (d-|z|))``; its coefficients estimate the Shapley
    values.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float).ravel()
    d = x.shape[0]

    sizes = rng.integers(1, d, size=n_samples)
    masks = np.zeros((n_samples, d), dtype=float)
    for i, s in enumerate(sizes):
        masks[i, rng.choice(d, size=s, replace=False)] = 1.0

    bg = background[rng.integers(0, background.shape[0], size=n_samples)]
    Z = np.where(masks > 0, x[None, :], bg)
    y = predict_proba(Z)[:, target]

    from scipy.special import comb
    s = masks.sum(axis=1)
    denom = comb(d, s) * s * (d - s)
    weights = np.where(denom > 0, (d - 1) / np.clip(denom, 1e-12, None), 0.0)
    weights = np.clip(weights, 0, 1e6)

    return _weighted_linear_coefs(masks, y, weights)


def _weighted_linear_coefs(X: np.ndarray, y: np.ndarray, w: np.ndarray
                           ) -> np.ndarray:
    """Weighted least squares; return coefficients (no intercept term)."""
    X1 = np.hstack([np.ones((X.shape[0], 1)), X])
    W = np.sqrt(np.clip(w, 0, None))[:, None]
    Xw, yw = X1 * W, y * W.ravel()
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef[1:]


# --------------------------------------------------------------------------- #
# Gradient family (require a torch model)
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


def vanilla_gradient(model, x: np.ndarray, target: int) -> np.ndarray:
    r"""d f_target / d x  -- the raw input gradient (Simonyan et al., 2014)."""
    torch = _torch()
    model.eval()
    xt = torch.tensor(np.asarray(x, dtype=np.float32)[None], requires_grad=True)
    out = model(xt)
    out[0, target].backward()
    return xt.grad.detach().numpy().ravel()


def gradient_x_input(model, x: np.ndarray, target: int) -> np.ndarray:
    r"""Gradient * Input (Shrikumar et al., 2017)."""
    g = vanilla_gradient(model, x, target)
    return g * np.asarray(x, dtype=float).ravel()


def integrated_gradients(model, x: np.ndarray, target: int,
                         baseline: np.ndarray | None = None,
                         steps: int = 50) -> np.ndarray:
    r"""Integrated Gradients (Sundararajan et al., 2017), zero baseline default."""
    torch = _torch()
    model.eval()
    x = np.asarray(x, dtype=np.float32).ravel()
    baseline = np.zeros_like(x) if baseline is None else np.asarray(baseline, np.float32).ravel()
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
    path = np.stack([baseline + a * (x - baseline) for a in alphas]).astype(np.float32)
    xt = torch.tensor(path, requires_grad=True)
    out = model(xt)
    grads = torch.autograd.grad(out[:, target].sum(), xt)[0].detach().numpy()
    avg_grad = grads.mean(axis=0)
    return (x - baseline) * avg_grad


def smoothgrad(model, x: np.ndarray, target: int, n_samples: int = 50,
               noise: float = 0.15, rng: np.random.Generator | None = None,
               times_input: bool = False) -> np.ndarray:
    r"""SmoothGrad (Smilkov et al., 2017): gradient averaged over Gaussian noise.

    With ``times_input=True`` the noise-averaged gradient is multiplied by the
    input, giving the "SmoothGrad * Input" attribution. This places SmoothGrad on
    the same ``x * grad`` footing as Integrated Gradients, so the two gradient
    methods share a sign basis and cluster as one family (poster Finding 3).
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float).ravel()
    sigma = noise * (x.max() - x.min() + 1e-12)
    acc = np.zeros_like(x)
    for _ in range(n_samples):
        acc += vanilla_gradient(model, x + rng.normal(0, sigma, size=x.shape), target)
    grad = acc / n_samples
    return grad * x if times_input else grad


# --------------------------------------------------------------------------- #
# Registry -- the four methods the poster uses, in a fixed display order
# --------------------------------------------------------------------------- #
POSTER_METHODS = ["LIME", "KernelSHAP", "IG", "SmoothGrad"]
METHOD_FAMILY = {
    "LIME": "perturbation",
    "KernelSHAP": "perturbation",
    "Grad": "gradient",
    "Grad*Input": "gradient",
    "IG": "gradient",
    "SmoothGrad": "gradient",
}
