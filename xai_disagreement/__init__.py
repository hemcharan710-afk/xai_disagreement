"""
xai_disagreement
================

Shared core library for the XAI *disagreement problem* studies.

This package is the faithful re-implementation of the quantitative framework from

    Krishna, Han, Gu, Wu, Jabbari, Lakkaraju (2024).
    "The Disagreement Problem in Explainable Machine Learning: A Practitioner's
     Perspective." Transactions on Machine Learning Research.

and is reused by both of our own studies (the poster
"The Disagreement Problem: How Model Generalization Impacts XAI Consensus"):

    * adult_income_study/  -- the original poster experiment (UCI Adult Income + MLPs)
    * mnist_cnn_study/     -- the image-domain extension (MNIST + a basic CNN)

Three building blocks:
    metrics     -- the six paper metrics + our RA / SA / SRA + predictive entropy
    explainers  -- LIME, KernelSHAP, VanillaGradient, Gradient*Input,
                   IntegratedGradients, SmoothGrad
    alignment   -- project every attribution onto one common feature index space
                   (so "index mismatch" is removed as a confound, exactly as the
                   poster describes)
"""

from . import metrics, explainers, alignment, aggregate, utils  # noqa: F401

__all__ = ["metrics", "explainers", "alignment", "aggregate", "utils"]
__version__ = "1.0.0"
