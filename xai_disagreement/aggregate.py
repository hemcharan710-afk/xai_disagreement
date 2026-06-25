r"""
Aggregation across method pairs and instances
=============================================

Every figure in the poster reduces a stack of per-instance attributions to one
of three shapes:

    * a method x method **agreement matrix** (Fig. 1b, Fig. 3 dendrogram input)
    * a single **mean pairwise score** per model            (Finding 1 decay)
    * a per-instance **mean pairwise score** vector          (Finding 2 paradox)

This module computes all three from one common input:

    ``attrs``  -- dict ``method_name -> ndarray (n_instances, n_features)`` of
                  signed, index-aligned attributions.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import metrics as M


def method_pairs(methods: list[str]) -> list[tuple[str, str]]:
    return list(combinations(methods, 2))


def _metric_fn(name: str):
    """Resolve a metric name to ``fn(a, b, k)`` (k ignored where not needed)."""
    if name in M.PAPER_TOPK_METRICS:
        fn = M.PAPER_TOPK_METRICS[name]
        return lambda a, b, k: fn(a, b, k)
    if name == "rank_correlation":
        return lambda a, b, k: M.rank_correlation(a, b)
    if name == "pairwise_rank_agreement":
        return lambda a, b, k: M.pairwise_rank_agreement(a, b)
    if name == "RA":
        return lambda a, b, k: M.poster_rank_agreement(a, b)
    if name == "SA":
        return lambda a, b, k: M.poster_sign_agreement(a, b)
    if name == "SRA":
        return lambda a, b, k: M.poster_signed_rank_agreement(a, b, k)
    raise KeyError(name)


def per_instance_pair_scores(attrs: dict[str, np.ndarray], metric: str, k: int
                             ) -> dict[tuple[str, str], np.ndarray]:
    """For each method pair, the metric value at every instance (shape n)."""
    fn = _metric_fn(metric)
    methods = list(attrs)
    n = next(iter(attrs.values())).shape[0]
    out: dict[tuple[str, str], np.ndarray] = {}
    for ma, mb in method_pairs(methods):
        out[(ma, mb)] = np.array(
            [fn(attrs[ma][i], attrs[mb][i], k) for i in range(n)]
        )
    return out


def agreement_matrix(attrs: dict[str, np.ndarray], metric: str, k: int
                     ) -> tuple[np.ndarray, list[str]]:
    """Symmetric method x method matrix of the instance-mean metric (diag=1)."""
    methods = list(attrs)
    pair_scores = per_instance_pair_scores(attrs, metric, k)
    m = len(methods)
    mat = np.eye(m)
    for (ma, mb), vals in pair_scores.items():
        i, j = methods.index(ma), methods.index(mb)
        mat[i, j] = mat[j, i] = float(np.mean(vals))
    return mat, methods


def mean_pairwise(attrs: dict[str, np.ndarray], metric: str, k: int) -> float:
    """Single number: metric averaged over all method pairs and instances."""
    pair_scores = per_instance_pair_scores(attrs, metric, k)
    return float(np.mean([v.mean() for v in pair_scores.values()]))


def per_instance_mean(attrs: dict[str, np.ndarray], metric: str, k: int
                      ) -> np.ndarray:
    """Per-instance mean over method pairs (shape n) -- the paradox x-axis pair."""
    pair_scores = per_instance_pair_scores(attrs, metric, k)
    return np.mean(np.stack(list(pair_scores.values())), axis=0)
