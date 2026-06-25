"""
Reproduce the Krishna et al. (2024) disagreement analysis on COMPAS.

    python -m paper_reproduction.run_disagreement [--k 5] [--instances 200]

Trains a neural network (and logistic regression) on COMPAS, applies all SIX
explanation methods (LIME, KernelSHAP, Vanilla Gradient, Gradient*Input,
Integrated Gradients, SmoothGrad), and reproduces the paper's Figure-1-style
six-panel heatmap (one panel per metric) plus a rank-agreement-vs-k sweep
(Figure 2). Outputs go to ``paper_reproduction/figures/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xai_disagreement import explainers as E, aggregate as AG  # noqa: E402
from xai_disagreement.utils import set_seed  # noqa: E402
from paper_reproduction import data as D, models as Mod  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures")

# the paper's six methods, in display order
METHODS = ["LIME", "KernelSHAP", "Grad", "Grad*Input", "IG", "SmoothGrad"]
# the paper's six metrics (Figure 1 panels)
METRICS = ["rank_correlation", "pairwise_rank_agreement", "feature_agreement",
           "rank_agreement", "sign_agreement", "signed_rank_agreement"]
METRIC_TITLE = {
    "rank_correlation": "Rank correlation",
    "pairwise_rank_agreement": "Pairwise rank agreement",
    "feature_agreement": "Feature agreement",
    "rank_agreement": "Rank agreement",
    "sign_agreement": "Sign agreement",
    "signed_rank_agreement": "Signed rank agreement",
}


def explain_all_six(model, ds, instances, background, n_perturb, rng):
    """method -> (n_instances, n_features) signed attributions for all 6 methods."""
    nf = ds.n_features
    preds = model.predict_proba(instances).argmax(1)
    pp = model.predict_proba
    out = {m: np.zeros((len(instances), nf)) for m in METHODS}
    for i, x in enumerate(instances):
        t = int(preds[i])
        out["LIME"][i] = E.lime(x, pp, t, background, n_samples=n_perturb, rng=rng)
        out["KernelSHAP"][i] = E.kernel_shap(x, pp, t, background,
                                             n_samples=n_perturb, rng=rng)
        out["Grad"][i] = E.vanilla_gradient(model, x, t)
        out["Grad*Input"][i] = E.gradient_x_input(model, x, t)
        out["IG"][i] = E.integrated_gradients(model, x, t, steps=50)
        out["SmoothGrad"][i] = E.smoothgrad(model, x, t, n_samples=50, rng=rng)
    return out


def six_panel_heatmap(attrs, k, title, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, metric in zip(axes.ravel(), METRICS):
        mat, methods = AG.agreement_matrix(attrs, metric, k)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Reds")
        ax.set_xticks(range(len(methods))); ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(methods, fontsize=8)
        for i in range(len(methods)):
            for j in range(len(methods)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7)
        suffix = f" (k={k})" if metric in ("feature_agreement", "rank_agreement",
                                           "sign_agreement", "signed_rank_agreement") else ""
        ax.set_title(METRIC_TITLE[metric] + suffix, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)


def rank_agreement_vs_k(attrs, ks, title, fname):
    pairs = AG.method_pairs(METHODS)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ma, mb in pairs:
        ys = [float(np.mean(AG.per_instance_pair_scores(
            attrs, "rank_agreement", k)[(ma, mb)])) for k in ks]
        ax.plot(ks, ys, marker="o", markersize=3, linewidth=1, alpha=0.8,
                label=f"{ma}-{mb}")
    ax.set_xlabel("k (top-k features)"); ax.set_ylabel("Rank agreement")
    ax.set_title(title); ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--n-perturb", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    ds = D.load_compas(seed=args.seed)
    print(f"COMPAS: {ds.n_features} features, "
          f"{len(ds.X_train)} train / {len(ds.X_test)} test")

    nn_model = Mod.train(Mod.TorchMLP(ds.n_features), ds, seed=args.seed)
    lr_model = Mod.train(Mod.TorchLogReg(ds.n_features), ds, epochs=100,
                         seed=args.seed)
    print(f"  NN  test acc {Mod.accuracy(nn_model, ds.X_test, ds.y_test):.3f}")
    print(f"  LR  test acc {Mod.accuracy(lr_model, ds.X_test, ds.y_test):.3f}")

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(ds.X_test), size=min(args.instances, len(ds.X_test)),
                     replace=False)
    instances = ds.X_test[sel]
    background = ds.X_train[rng.choice(len(ds.X_train), 200, replace=False)]

    summary = {}
    for name, model in (("neural_network", nn_model), ("logistic_regression", lr_model)):
        attrs = explain_all_six(model, ds, instances, background,
                                args.n_perturb, rng)
        six_panel_heatmap(attrs, args.k,
                          f"COMPAS {name.replace('_', ' ')} -- "
                          f"six disagreement metrics (k={args.k})",
                          f"fig1_{name}_metrics.png")
        if name == "neural_network":
            rank_agreement_vs_k(attrs, list(range(1, ds.n_features + 1)),
                                "COMPAS NN -- rank agreement vs k (Fig. 2 style)",
                                "fig2_rank_agreement_vs_k.png")
        summary[name] = {
            metric: AG.agreement_matrix(attrs, metric, args.k)[0].tolist()
            for metric in METRICS}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"methods": METHODS, "k": args.k, "matrices": summary}, f, indent=2)
    print(f"\nFigures + results.json written to {OUT}/")


if __name__ == "__main__":
    main()
