"""
Generate the four poster attributions for a trained Adult model.

All four methods return one signed value per one-hot column, so they already
share the same 105-dim index space -- alignment is the identity here (the
``align_tabular`` shape-check still runs, matching the poster's stated control).
"""
from __future__ import annotations

import numpy as np

from xai_disagreement import explainers as E, alignment as A


def explain_model(trained, ds, instances: np.ndarray, background: np.ndarray,
                  n_perturb: int = 800, ig_steps: int = 32, sg_samples: int = 25,
                  seed: int = 0) -> dict[str, np.ndarray]:
    """Return ``method -> (n_instances, n_features)`` signed attributions.

    Attributions explain each instance's *predicted* class, so disagreement is
    measured about the decision the model actually makes.
    """
    model = trained.model
    rng = np.random.default_rng(seed)
    nf = ds.n_features
    preds = model.predict_proba(instances).argmax(1)
    pp = model.predict_proba

    out = {m: np.zeros((len(instances), nf)) for m in E.POSTER_METHODS}
    for i, x in enumerate(instances):
        t = int(preds[i])
        out["LIME"][i] = A.align_tabular(
            E.lime(x, pp, t, background, n_samples=n_perturb, rng=rng), nf)
        out["KernelSHAP"][i] = A.align_tabular(
            E.kernel_shap(x, pp, t, background, n_samples=n_perturb, rng=rng), nf)
        out["IG"][i] = A.align_tabular(
            E.integrated_gradients(model, x, t, steps=ig_steps), nf)
        out["SmoothGrad"][i] = A.align_tabular(
            E.smoothgrad(model, x, t, n_samples=sg_samples, rng=rng,
                         times_input=True), nf)
    return out
