# Adult Income study — the poster experiment

Our **original** study, exactly as on the poster: does *model generalization
quality* drive XAI consensus, independent of mathematical alignment?

* **Dataset** — UCI Adult Income, binary (`income > 50K`), one-hot encoded to
  ~105 columns (fetched once from OpenML, cached under `data/`).
* **Model spectrum** (one MLP trained four ways, best → worst generalization):

  | | Variant | Corruption | Role |
  |---|---|---|---|
  | **A** | Control | none | baseline (high gen.) |
  | **C** | Feature Noise | mild input noise + 8 % label flips | mild degradation |
  | **D** | Label Poison | 20 % label flips | severe degradation |
  | **B** | Overfit | big unregularized net fitting 34 % label noise | worst generalization |

  The corruption rises monotonically A→C→D→B so that *generalization* (test
  accuracy) — and with it the consensus measured in Finding 1 — degrades in
  order. The named mechanism of each variant is preserved; the increasing label
  corruption underneath is what scrambles the decision boundary and drives the
  monotone SRA decline (a memorized *small subset* was avoided on purpose — it
  makes the net flat on test points, which inflates agreement instead).

* **Methods (4)** — LIME, KernelSHAP, IG, SmoothGrad, all on the same ~103
  feature indices. The two gradient methods share an `x·grad` basis (IG, and
  SmoothGrad×Input) so they cluster as one family in Finding 3.
* **Metrics** — RA (magnitude Spearman), SA (sign match), **SRA** (signed-rank
  correlation, primary). See [`../docs/method_notes.md`](../docs/method_notes.md).

## Run

```bash
python -m adult_income_study.run_all --instances 150     # full
python -m adult_income_study.run_all --quick             # fast smoke test
```

## Outputs (`figures/`)

* `finding1_consensus_decay.png` — mean pairwise RA/SA/SRA across A,C,D,B.
* `finding2_uncertainty_paradox.png` — entropy vs SRA, with per-model Pearson r.
* `finding3_families.png` — per-model agreement dendrograms.
* `finding3_pairwise_bars.png` — rank agreement per method pair across models.
* `heatmaps_SRA.png`, `results.json`.

See the top-level [README](../README.md#reproducibility-caveat) for why absolute
numbers differ from the poster.
