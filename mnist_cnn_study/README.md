# MNIST + CNN study — image-domain extension

The poster's *Future Work* asks to "extend to image/NLP datasets." This applies
the exact same study design to **MNIST with a basic CNN**.

* **Dataset** — MNIST (subsampled, cached under `data/` via torchvision),
  flat 784-vectors in [0,1].
* **Model** — a basic CNN (`conv16 → conv32 → fc`), trained four ways across the
  same generalization spectrum:

  | Variant | Corruption | Role |
  |---|---|---|
  | **A** Control | none | high gen. |
  | **C** Feature Noise | Gaussian pixel noise | low gen. |
  | **D** Label Poison | 20 % labels randomized | low gen. |
  | **B** Overfit | large head, no reg, memorize a subset | worst gen. |

* **Methods (4)** — LIME, KernelSHAP, IG, SmoothGrad.
* **Alignment** — the image-domain control: LIME/KernelSHAP act on a **7×7
  super-pixel grid** (49 patches); IG/SmoothGrad per-pixel attributions are
  mean-pooled into the *same* 49 patches, so all four methods share one index
  space. See `explain.py` and `alignment.align_to_superpixels`.

## Run

```bash
python -m mnist_cnn_study.run_all --instances 120        # full
python -m mnist_cnn_study.run_all --quick                # fast smoke test
```

## Outputs (`figures/`)

Same four findings figures + `heatmaps_SRA.png` + `results.json` as the Adult
study. In a sample run all four models showed a **positive** entropy↔SRA
correlation (Finding 2 / the Uncertainty Paradox), consistent with the poster.
