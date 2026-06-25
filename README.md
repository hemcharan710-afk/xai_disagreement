# The Disagreement Problem in Explainable ML

Research code for **"The Disagreement Problem: How Model Generalization
Impacts XAI Consensus"**
(Kowel P Laloo · Chitiveli Hemcharan Varma · advisor Prof. Manisha Padala,
Dept. of CSE, IIT Gandhinagar).

We ask a question the original disagreement literature leaves open: is XAI
disagreement only a property of the *explanation methods*, or also a symptom of
**model quality**? We build a from-scratch framework, run it across a
generalization spectrum on two datasets, and report what we find.

The work is layered on top of the framework introduced by:

> Krishna, Han, Gu, Wu, Jabbari, Lakkaraju (2024).
> *The Disagreement Problem in Explainable Machine Learning: A Practitioner's
> Perspective.* Transactions on Machine Learning Research.

Two artifacts live alongside the code:
[`2202.01602v6-4.pdf`](./) (the framework paper) and
[`xai_disagreement_poster (1).pdf`](./xai_disagreement_poster%20(1).pdf) (our poster).

---

## What's here

```
xai_disagreement/        Shared core library (the disagreement framework)
  metrics.py             6 base metrics + weighted rank agr. + our RA/SA/SRA + entropy
  explainers.py          LIME, KernelSHAP, VanillaGrad, Grad*Input, IG, SmoothGrad
  alignment.py           project every attribution onto one shared index space
  aggregate.py           average metrics over method pairs and instances
  findings.py            the 3 findings as reusable analysis + plots

paper_reproduction/      Framework check on COMPAS (6 methods, 6 metrics)
adult_income_study/      OUR main study: UCI Adult Income + 4 MLPs  (the poster)
mnist_cnn_study/         OUR extension: MNIST + a basic CNN

docs/method_notes.md     Formal definitions of every metric
```

We build the framework first (`xai_disagreement/`), sanity-check it on COMPAS
(`paper_reproduction/`), then run our own research — `adult_income_study/` is the
headline study, `mnist_cnn_study/` extends it to images.

---

## Setup

The repo is self-contained — LIME, KernelSHAP and the gradient methods are all
implemented from scratch, so no `lime`/`shap`/`captum` is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> A `.venv/` built with `numpy<2` already exists in this folder; the global
> Anaconda environment on this machine has a broken numpy 1.x/2.x ABI mix and is
> **not** used.

---

## Run

```bash
# 1. Framework check (COMPAS, 6 methods, 6 metrics -> heatmaps)
python -m paper_reproduction.run_disagreement

# 2. Main study (Adult Income, 4 MLP variants, 3 findings)
python -m adult_income_study.run_all              # add --quick for a fast smoke run

# 3. Extension (MNIST + CNN, 3 findings)
python -m mnist_cnn_study.run_all                 # add --quick for a fast smoke run
```

Each writes PNG figures + a `results.json` to its own `figures/` folder.

---

## The study in one paragraph

Prior work treats XAI disagreement as a mathematical artifact of the *methods*.
Our hypothesis: it is also a symptom of **model quality** — poorly generalizing
models produce unreliable gradient landscapes that amplify or mask inter-method
divergence. We train one architecture four ways across a generalization
spectrum — **A** Control · **C** Feature-Noise · **D** Label-Poison · **B**
Overfit — apply four XAI methods (LIME, KernelSHAP, IG, SmoothGrad) all
projected onto the same feature indices, and measure pairwise Rank Agreement
(RA), Sign Agreement (SA) and Signed Rank Agreement (SRA, primary).

---

## Results

**Bottom line:** across two datasets, model quality leaves a measurable mark on
XAI consensus, but it is a weak signal — agreement drifts down as a model
overfits, "more agreement" does not mean "better explanation," and methods split
into families that agree with their own kind. Details below.

### Models trained (accuracy / generalization gap)

**Adult Income (MLP), k = 10**

| Variant | Train acc | Test acc | Gen. gap |
|---|---|---|---|
| A — Control | 0.884 | 0.836 | 0.048 |
| C — Feature-Noise | 0.846 | 0.834 | 0.012 |
| D — Label-Poison | 0.817 | 0.789 | 0.028 |
| B — Overfit | 0.848 | 0.797 | 0.051 |

**MNIST (CNN), k = 8**

| Variant | Train acc | Test acc | Gen. gap |
|---|---|---|---|
| A — Control | 0.974 | 0.963 | 0.011 |
| C — Feature-Noise | 0.929 | 0.933 | −0.004 |
| D — Label-Poison | 0.965 | 0.957 | 0.008 |
| B — Overfit | 0.961 | 0.958 | 0.003 |

*In short:* on Adult Income the gaps behave as designed — the Overfit model (B)
has the largest train/test gap and Feature-Noise (C) the smallest. On MNIST even
the "overfit" CNN generalizes well, so every gap is tiny — which is why the MNIST
effects below come out smaller.

### Finding 1 — Consensus tracks generalization

Mean pairwise SRA is highest for the well-generalizing controls and lowest for
the overfit model on both datasets.

| Variant | Adult SRA | MNIST SRA |
|---|---|---|
| A — Control | 0.193 | 0.329 |
| C — Feature-Noise | 0.201 | 0.350 |
| D — Label-Poison | 0.187 | 0.343 |
| B — Overfit | 0.190 | 0.309 |

![Finding 1 — SRA across the generalization spectrum](figures/finding1_sra.png)

Control → Overfit SRA decline: **1.8%** (Adult), **6.1%** (MNIST). The effect is
real but modest, and not strictly monotonic — the feature-noise variant (C)
edges out the control, which we read as label noise hurting consensus more than
input noise.

*In short:* explanation methods agree most on a healthy model and least on an
overfit one — evidence that disagreement is partly about the model, not just the
methods.

### Finding 2 — The Uncertainty Paradox

Predictive entropy and SRA are *not* consistently positively correlated;
degraded models give flat, near-zero attributions, so methods can trivially
"agree on nothing." Pearson r (entropy vs SRA):

| Variant | Adult | MNIST |
|---|---|---|
| A | −0.14 | 0.01 |
| C | 0.11 | 0.01 |
| D | −0.19 | −0.19 |
| B | −0.07 | 0.17 |

![Finding 2 — entropy vs SRA correlation per variant](figures/finding2_uncertainty.png)

The sign flips across variants — high agreement is not evidence of an
informative explanation.

*In short:* a degraded model can make methods "agree" simply because all their
attributions collapse toward zero. Consensus can be a sign of a broken model, not
a trustworthy one.

### Finding 3 — Family-structured disagreement

Sign Agreement is dominated by a single pairing on both datasets: KernelSHAP and
SmoothGrad agree far more than the gradient/perturbation cross-pairs.

| Pair (mean SA) | Adult | MNIST |
|---|---|---|
| KernelSHAP–SmoothGrad | ~0.50 | ~0.54 |
| IG–SmoothGrad | ~0.08 | ~0.31 |
| LIME–IG | ~0.10 | ~0.31 |

![Finding 3 — sign agreement by method pair](figures/finding3_families.png)

On COMPAS the framework check shows the same family structure: gradient methods
cluster tightly (Grad–SmoothGrad SRA 0.60) and LIME–KernelSHAP cluster (SRA
0.52), confirming the metric implementations behave as expected before we apply
them to our own models.

*In short:* methods agree mostly with others of the same type (gradient-based vs.
perturbation-based), so within-family agreement reflects shared assumptions, not
proof that the explanation is correct.

---

## Notes on method

LIME, KernelSHAP and the gradient explainers are implemented from scratch, so
absolute metric values depend on our samplers and seeds rather than any external
library. Everything in the Results section is computed from freshly trained
models; rerunning `*.run_all` regenerates every figure and `results.json`. See
[`docs/method_notes.md`](docs/method_notes.md) for exact metric definitions and
the one adaptation we make to the SRA formula.
