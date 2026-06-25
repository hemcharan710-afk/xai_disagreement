# Paper reproduction — Krishna et al. (2024) on COMPAS

Faithful reproduction of the original paper's quantitative framework, built
**first** so the two studies can reuse it.

* **Dataset** — COMPAS, 7 features, predict high vs low risk
  (`decile_score ≥ 5`). ProPublica's standard filters; 4,937 training defendants,
  matching the paper.
* **Models** — neural network (`50-100-50`) and logistic regression, both in
  torch (gradients apply to both, as in the paper).
* **Methods (6)** — LIME, KernelSHAP, Vanilla Gradient, Gradient×Input,
  Integrated Gradients, SmoothGrad.
* **Metrics (6)** — feature / rank / sign / signed-rank agreement (top-k, k=5),
  rank correlation and pairwise rank agreement (all features).

## Run

```bash
python -m paper_reproduction.run_disagreement --k 5 --instances 200
```

## Outputs (`figures/`)

* `fig1_neural_network_metrics.png` — the paper's Figure-1 six-panel heatmap.
* `fig1_logistic_regression_metrics.png` — same for LR.
* `fig2_rank_agreement_vs_k.png` — rank agreement as k grows (paper Fig. 2).
* `results.json` — every agreement matrix.

## What to look for

The paper's headline pattern should reappear: gradient-method pairs
(Grad–Grad×Input, Grad–IG, IG–SmoothGrad…) show **stronger** agreement, while
pairs crossing the perturbation/gradient divide disagree more; stricter metrics
(signed rank agreement) are uniformly lower than lenient ones (feature
agreement); and agreement falls as k increases.

Sanity check from a sample run: NN test acc ≈ 0.747, LR ≈ 0.757 (paper: 0.73,
0.75).
