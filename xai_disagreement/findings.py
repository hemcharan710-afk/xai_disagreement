r"""
The three poster findings, as reusable analysis + plotting routines
==================================================================

Both studies (Adult Income, MNIST) feed the same structure in and get the same
three figures + a results dict out:

    Finding 1  Overfitting destroys consensus
               mean pairwise RA / SA / SRA, ordered best->worst generalization;
               expect a monotone decline.

    Finding 2  The Uncertainty Paradox
               per-instance predictive entropy vs per-instance mean pairwise SRA;
               a *positive* slope means degraded, high-entropy predictions show
               spuriously high agreement (flat gradient landscapes).

    Finding 3  Family-structured disagreement
               method x method agreement + a dendrogram per model; gradient
               methods (IG, SmoothGrad) cluster, perturbation LIME stays isolated.

Input ``results`` is an ordered dict ``variant -> ModelResult`` where the
variants are listed best->worst generalization (e.g. A, C, D, B).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from . import aggregate as AG, metrics as M  # noqa: E402


@dataclass
class ModelResult:
    variant: str
    label: str
    attrs: dict[str, np.ndarray]   # method -> (n_instances, n_features)
    proba: np.ndarray              # (n_instances, n_classes)
    train_acc: float
    test_acc: float
    gen_gap: float


# --------------------------------------------------------------------------- #
def finding1_consensus_decay(results: "dict[str, ModelResult]", k: int,
                             out_dir: str, title: str) -> dict:
    variants = list(results)
    labels = [results[v].label for v in variants]
    scores = {m: [] for m in ("RA", "SA", "SRA")}
    for v in variants:
        for m in scores:
            scores[m].append(AG.mean_pairwise(results[v].attrs, m, k))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(len(variants))
    for m, mk in zip(("RA", "SA", "SRA"), ("o-", "s--", "^-")):
        axl.plot(xs, scores[m], mk, label=m, linewidth=2)
    axl.set_xticks(xs); axl.set_xticklabels(labels, rotation=15)
    axl.set_ylabel("Agreement score"); axl.set_ylim(0, 1)
    axl.set_title("Fig. 1a  Consensus decay (better -> worse generalization)")
    axl.legend(); axl.grid(alpha=0.3)

    w = 0.25
    for i, m in enumerate(("RA", "SA", "SRA")):
        axr.bar(xs + (i - 1) * w, scores[m], w, label=m)
    axr.set_xticks(xs); axr.set_xticklabels(labels, rotation=15)
    axr.set_ylabel("Agreement score"); axr.set_ylim(0, 1)
    axr.set_title("Fig. 1b  RA / SA / SRA per model"); axr.legend()
    fig.suptitle(f"{title} -- Finding 1: Overfitting Destroys Consensus")
    fig.tight_layout()
    _save(fig, out_dir, "finding1_consensus_decay.png")

    sra = scores["SRA"]
    decline = (sra[0] - sra[-1]) / sra[0] * 100 if sra[0] else float("nan")
    return {"variants": variants, "labels": labels, "scores": scores,
            "SRA_first": sra[0], "SRA_last": sra[-1],
            "SRA_relative_decline_pct": decline}


# --------------------------------------------------------------------------- #
def finding2_uncertainty_paradox(results: "dict[str, ModelResult]", k: int,
                                 out_dir: str, title: str) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    corr = {}
    for c, (v, r) in zip(colors, results.items()):
        ent = M.predictive_entropy(r.proba)
        sra = AG.per_instance_mean(r.attrs, "SRA", k)
        if np.ptp(ent) > 0 and np.ptp(sra) > 0:
            rho = float(pearsonr(ent, sra)[0])
            b, a = np.polyfit(ent, sra, 1)
        else:
            rho, b, a = 0.0, 0.0, float(np.mean(sra))
        corr[v] = rho
        ax.scatter(ent, sra, s=10, alpha=0.25, color=c)
        xs = np.linspace(ent.min(), ent.max(), 20)
        ax.plot(xs, b * xs + a, color=c, linewidth=2.5,
                label=f"{r.label}  r={rho:+.2f}")
    ax.set_xlabel("Predictive entropy H(P)")
    ax.set_ylabel("Mean pairwise signed rank agreement (SRA)")
    ax.set_title(f"{title} -- Finding 2: The Uncertainty Paradox")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "finding2_uncertainty_paradox.png")
    return {"entropy_sra_pearson_r": corr}


# --------------------------------------------------------------------------- #
def finding3_families(results: "dict[str, ModelResult]", k: int,
                      out_dir: str, title: str,
                      cluster_metric: str = "RA") -> dict:
    variants = list(results)
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    pair_table = {}
    for ax, v in zip(axes, variants):
        mat, methods = AG.agreement_matrix(results[v].attrs, cluster_metric, k)
        dist = np.clip(1.0 - mat, 0, None)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        dendrogram(Z, labels=methods, ax=ax, color_threshold=0.0,
                   above_threshold_color="C0")
        ax.set_title(results[v].label, fontsize=10)
        ax.set_ylabel(f"distance (1 - {cluster_metric})")
        # report the three poster pairs via sign agreement
        sa, _ = AG.agreement_matrix(results[v].attrs, "SA", k)
        idx = {m: i for i, m in enumerate(methods)}
        pair_table[v] = {
            "IG-SmoothGrad_SA": _safe(sa, idx, "IG", "SmoothGrad"),
            "KernelSHAP-SmoothGrad_SA": _safe(sa, idx, "KernelSHAP", "SmoothGrad"),
            "LIME-IG_SA": _safe(sa, idx, "LIME", "IG"),
        }
    fig.suptitle(f"{title} -- Finding 3: Family-Structured Disagreement "
                 f"(dendrograms on 1 - {cluster_metric})")
    fig.tight_layout()
    _save(fig, out_dir, "finding3_families.png")

    _plot_pairwise_bars(results, k, out_dir, title)
    return {"sign_agreement_pairs": pair_table}


def _plot_pairwise_bars(results, k, out_dir, title):
    """Per-method-pair RA across models (poster's bottom-left bar chart)."""
    variants = list(results)
    pairs = AG.method_pairs(list(next(iter(results.values())).attrs))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(variants))
    w = 0.8 / len(pairs)
    for j, (ma, mb) in enumerate(pairs):
        vals = [float(np.mean(AG.per_instance_pair_scores(
            results[v].attrs, "RA", k)[(ma, mb)])) for v in variants]
        ax.bar(xs + (j - len(pairs) / 2) * w, vals, w, label=f"{ma} vs {mb}")
    ax.set_xticks(xs); ax.set_xticklabels([results[v].label for v in variants],
                                          rotation=15)
    ax.set_ylabel("Rank agreement (RA)")
    ax.set_title(f"{title} -- Rank agreement per method pair across models")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "finding3_pairwise_bars.png")


# --------------------------------------------------------------------------- #
def heatmaps(results: "dict[str, ModelResult]", k: int, out_dir: str,
             title: str, metric: str = "SRA") -> None:
    """Krishna-Fig-1-style method x method heatmap per model (one metric)."""
    variants = list(results)
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        mat, methods = AG.agreement_matrix(results[v].attrs, metric, k)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Reds")
        ax.set_xticks(range(len(methods))); ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(methods, fontsize=7)
        for i in range(len(methods)):
            for j in range(len(methods)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
        ax.set_title(results[v].label, fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(f"{title} -- {metric} agreement heatmaps")
    _save(fig, out_dir, f"heatmaps_{metric}.png")


# --------------------------------------------------------------------------- #
def run_all_findings(results: "dict[str, ModelResult]", k: int, out_dir: str,
                     title: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "title": title, "k": k,
        "models": {v: {"label": r.label, "train_acc": r.train_acc,
                       "test_acc": r.test_acc, "gen_gap": r.gen_gap}
                   for v, r in results.items()},
        "finding1": finding1_consensus_decay(results, k, out_dir, title),
        "finding2": finding2_uncertainty_paradox(results, k, out_dir, title),
        "finding3": finding3_families(results, k, out_dir, title),
    }
    heatmaps(results, k, out_dir, title, "SRA")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
def _safe(mat, idx, a, b):
    if a in idx and b in idx:
        return float(mat[idx[a], idx[b]])
    return None


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
