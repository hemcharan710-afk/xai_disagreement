"""
Run the poster study on MNIST with a basic CNN (image-domain extension).

    python -m mnist_cnn_study.run_all [--instances 120] [--quick]

Same design as the Adult study: four-model degradation spectrum, four XAI
methods aligned to a 7x7 super-pixel grid, the three poster findings written to
``mnist_cnn_study/figures/``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xai_disagreement import findings  # noqa: E402
from xai_disagreement.findings import ModelResult  # noqa: E402
from xai_disagreement.utils import set_seed  # noqa: E402
from mnist_cnn_study import data as D, models as Mod, explain as X  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=120)
    ap.add_argument("--k", type=int, default=8, help="top-k for top-k metrics")
    ap.add_argument("--grid", type=int, default=7, help="super-pixel grid (g x g)")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        args.instances, args.n_train, n_perturb = 24, 3000, 250
    else:
        n_perturb = 600

    set_seed(args.seed)
    ds = D.load_mnist(n_train=args.n_train, seed=args.seed)
    print(f"MNIST: {len(ds.X_train)} train / {len(ds.X_test)} test, "
          f"grid {args.grid}x{args.grid} = {args.grid**2} patches")

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(ds.X_test), size=min(args.instances, len(ds.X_test)),
                     replace=False)
    instances = ds.X_test[sel]

    results: dict[str, ModelResult] = {}
    for v in Mod.SPECTRUM:
        tm = Mod.train_variant(v, ds, seed=args.seed)
        print(f"  model {v} {Mod.VARIANT_LABEL[v]:18s} "
              f"train={tm.train_acc:.3f} test={tm.test_acc:.3f} "
              f"gap={tm.gen_gap:+.3f}")
        attrs, _ = X.explain_model(tm, ds, instances, grid=args.grid,
                                   n_perturb=n_perturb, seed=args.seed)
        results[v] = ModelResult(
            variant=v, label=Mod.VARIANT_LABEL[v], attrs=attrs,
            proba=tm.model.predict_proba(instances),
            train_acc=tm.train_acc, test_acc=tm.test_acc, gen_gap=tm.gen_gap)

    summary = findings.run_all_findings(results, args.k, OUT,
                                        title="MNIST (CNN)")
    f1, f2 = summary["finding1"], summary["finding2"]
    print("\n=== Findings ===")
    print("F1 mean-pairwise SRA: " +
          ", ".join(f"{v}={s:.3f}" for v, s in zip(results, f1["scores"]["SRA"])) +
          f"  (relative decline {f1['SRA_relative_decline_pct']:.1f}%)")
    print("F2 entropy<->SRA Pearson r per model: " +
          ", ".join(f"{v}={r:+.2f}" for v, r in f2["entropy_sra_pearson_r"].items()))
    print(f"\nFigures + results.json written to {OUT}/")


if __name__ == "__main__":
    main()
