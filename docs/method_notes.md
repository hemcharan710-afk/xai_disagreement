# Method notes — formal definitions

This file records the exact definitions implemented in
[`xai_disagreement/metrics.py`](../xai_disagreement/metrics.py), with their
source in Krishna et al. (2024), *The Disagreement Problem in Explainable
Machine Learning: A Practitioner's Perspective* (TMLR), and the adaptations our
poster makes.

Notation: two explanations `E_a`, `E_b` are signed attribution vectors over a
common feature set `F`. `TF(E,k)` is the top-`k` features of `E` by **magnitude**
`|attribution|`. `R(E,f)` is the rank of feature `f` (1 = most important, by
magnitude). `S(E,f)` is the sign of `f`. **Lower = stronger disagreement** for
every metric.

## Paper metrics — top-k (Appendix A.1)

| Metric | Definition |
|---|---|
| **Feature agreement** | `|TF(a,k) ∩ TF(b,k)| / k` |
| **Rank agreement** | fraction of top-k features shared *and* at the same rank |
| **Sign agreement** | fraction of top-k features shared *and* same sign |
| **Signed rank agreement** | shared *and* same sign *and* same rank (strictest) |
| **Weighted rank agreement** (D.2) | `(1/k) Σ_{f∈TF(a,k)∩TF(b,k)} (1 − |R(a,f)−R(b,f)|/k)` |

## Paper metrics — feature set `F` (Appendix A.2)

| Metric | Definition |
|---|---|
| **Rank correlation** | Spearman correlation `r_s(R(a,F), R(b,F))` |
| **Pairwise rank agreement** | fraction of pairs `(i<j)∈F` with the same relative ordering |

## Our poster's three aggregate scores

| Score | Definition | Notes |
|---|---|---|
| **RA** (Rank Agreement) | `max(0, r_s(|a|, |b|))` — Spearman correlation of the *magnitude* rankings, clipped to [0,1] | lowest of the three (magnitude orderings differ most across families) |
| **SA** (Sign Agreement) | fraction of *all* features with matching attribution sign | over the full feature set, not only top-k |
| **SRA** (Signed Rank Agreement, **primary**) | `(1 + r_s(a, b)) / 2` — Spearman correlation of the **signed** attributions, rescaled to [0,1] | see note below |

### Note on SRA

The poster writes SRA with a *hard* rank indicator
`1[rank_a(f)=rank_b(f)]`. For continuous attributions in a ~100-dim space, exact
integer-rank equality almost never holds, so the hard form collapses to ≈ 0 and
cannot reproduce the poster's reported magnitudes (≈ 0.65–0.73). We therefore
implement the **signed rank** form above: the Spearman correlation of the
*signed* attributions, rescaled to `[0, 1]`. Ranking the signed values rewards a
feature only when it has the same *direction* (sign) **and** comparable
*importance* (rank) in both explanations — exactly "combines rank and sign". It
is continuous, equals 1 for identical explanations, ≈ 0.5 for unrelated ones, and
reproduces the poster's regime (SRA > SA > RA, SRA in the 0.55–0.73 band). The
strict top-k indicator version is still available as
`metrics.signed_rank_agreement` and is what the paper reproduction uses.

## Predictive entropy (Finding 2)

`H(P) = −Σ_c p_c log p_c` (nats) over the model's class-probability output.
Finding 2 (the *Uncertainty Paradox*) correlates per-instance `H(P)` with the
per-instance mean pairwise SRA. The poster hypothesises a **positive** slope
(uncertain predictions → flat near-zero attributions → methods trivially "agree
on nothing"). In our re-run this slope is **weak and not consistently positive**
(see the top-level README): with the signed-rank SRA, flat attributions score
≈ 0.5 rather than high, and — because our degraded models have *both* higher
entropy *and* lower SRA (Finding 1) — the across-model relationship is if
anything mildly negative. We report the measured per-model `r` rather than the
hypothesised sign.

## Alignment (the poster's confound control)

All attributions are projected onto one shared feature-index space before any
metric is computed (eliminating index mismatch as a confound):

* **Tabular** — every method already returns one value per column; alignment is
  the identity with a shape check (`alignment.align_tabular`).
* **Image (MNIST)** — perturbation methods (LIME, KernelSHAP) act on a `g×g`
  super-pixel grid; gradient methods' per-pixel attributions are mean-pooled
  into the same grid (`alignment.align_to_superpixels`). All four methods end up
  in one `g²`-dim space.
