"""
COMPAS loader for the Krishna et al. (2024) reproduction.

Follows the paper's tabular setup: seven features predicting the COMPAS risk
group (low vs high). Categorical columns are label-encoded so the feature space
stays at exactly seven dimensions, matching the paper (top-k metrics use k=5 of
7 features in their Figure 1).

Raw CSV is the ProPublica ``compas-scores-two-years`` file, cached under
``paper_reproduction/data/``.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "data")
URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
       "master/compas-scores-two-years.csv")

FEATURES = ["age", "two_year_recid", "priors_count", "length_of_stay",
            "c_charge_degree", "sex", "race"]


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_compas(test_size: float = 0.2, seed: int = 0) -> Dataset:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs(CACHE, exist_ok=True)
    csv = os.path.join(CACHE, "compas-scores-two-years.csv")
    if not os.path.exists(csv):
        urllib.request.urlretrieve(URL, csv)
    df = pd.read_csv(csv)

    # standard COMPAS cleaning (ProPublica filters)
    df = df[(df["days_b_screening_arrest"] <= 30)
            & (df["days_b_screening_arrest"] >= -30)
            & (df["is_recid"] != -1)
            & (df["c_charge_degree"] != "O")
            & (df["score_text"] != "N/A")].copy()

    import pandas as _pd
    df["length_of_stay"] = (
        (_pd.to_datetime(df["c_jail_out"]) - _pd.to_datetime(df["c_jail_in"]))
        .dt.total_seconds() / 86400.0
    ).fillna(0.0)

    for col in ("c_charge_degree", "sex", "race"):
        df[col] = df[col].astype("category").cat.codes

    X = df[FEATURES].astype(float).to_numpy()
    # label: high COMPAS risk (decile_score >= 5) vs low
    y = (df["decile_score"].to_numpy() >= 5).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    return Dataset(Xtr, Xte, ytr, yte, list(FEATURES))
