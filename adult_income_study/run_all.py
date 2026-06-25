"""
Run the full poster experiment on UCI Adult Income.

    python -m adult_income_study.run_all [--instances 150] [--quick]

Trains the four-model degradation spectrum (A, C, D, B), generates LIME /
KernelSHAP / IG / SmoothGrad attributions for a sample of test instances, and
produces the three poster findings under ``adult_income_study/figures/``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# allow running as a script (python adult_income_study/run_all.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xai_disagreement import findings  # noqa: E402
from xai_disagreement.findings import ModelResult  # noqa: E402
from xai_disagreement.utils import set_seed  # noqa: E402
from adult_income_study import data as D, models as Mod, explain as X  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=150,
                    help="test instances to explain per model")
    ap.add_argument("--k", type=int, default=10, help="top-k for the metrics")
    ap.add_argument("--max-rows", type=int, default=12000,
                    help="subsample Adult for speed (None-like 0 = all)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run for smoke-testing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        args.instances, args.max_rows, n_perturb = 30, 4000, 300
    else:
        n_perturb = 800

    set_seed(args.seed)
    ds = D.load_adult(seed=args.seed,
                      max_rows=(args.max_rows or None))
    print(f"Adult: {ds.n_features} features, "
          f"{len(ds.X_train)} train / {len(ds.X_test)} test")

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(ds.X_test), size=min(args.instances, len(ds.X_test)),
                     replace=False)
    instances = ds.X_test[sel]
    background = ds.X_train[rng.choice(len(ds.X_train), size=200, replace=False)]

    results: dict[str, ModelResult] = {}
    for v in Mod.SPECTRUM:                       # A, C, D, B (best -> worst)
        tm = Mod.train_variant(v, ds, seed=args.seed)
        print(f"  model {v:1s} {Mod.VARIANT_LABEL[v]:18s} "
              f"train={tm.train_acc:.3f} test={tm.test_acc:.3f} "
              f"gap={tm.gen_gap:+.3f}")
        attrs = X.explain_model(tm, ds, instances, background,
                                n_perturb=n_perturb, seed=args.seed)
        results[v] = ModelResult(
            variant=v, label=Mod.VARIANT_LABEL[v], attrs=attrs,
            proba=tm.model.predict_proba(instances),
            train_acc=tm.train_acc, test_acc=tm.test_acc, gen_gap=tm.gen_gap)

    summary = findings.run_all_findings(results, args.k, OUT,
                                        title="Adult Income (MLP)")
    f1, f2 = summary["finding1"], summary["finding2"]
    print("\n=== Findings ===")
    print(f"F1 mean-pairwise SRA: " +
          ", ".join(f"{results[v].label.split(' - ')[0]}={s:.3f}"
                    for v, s in zip(results, f1["scores"]["SRA"])) +
          f"  (relative decline {f1['SRA_relative_decline_pct']:.1f}%)")
    print("F2 entropy<->SRA Pearson r per model: " +
          ", ".join(f"{v}={r:+.2f}" for v, r in f2["entropy_sra_pearson_r"].items()))
    print(f"\nFigures + results.json written to {OUT}/")


if __name__ == "__main__":
    main()
