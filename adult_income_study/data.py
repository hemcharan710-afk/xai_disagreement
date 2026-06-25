"""
UCI Adult Income loader (poster dataset).

Binary classification: predict whether income > 50K. Categorical columns are
one-hot encoded and numeric columns standardized, giving ~105 feature columns
(exactly the "105 features after one-hot encoding" reported on the poster).

Network: the raw data is fetched once from OpenML and cached under ``data/``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "data")


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


def load_adult(test_size: float = 0.2, seed: int = 0,
               max_rows: int | None = None) -> Dataset:
    """Load, encode and split the Adult Income dataset."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs(CACHE, exist_ok=True)
    cache_file = os.path.join(CACHE, "adult.csv.gz")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
    else:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml("adult", version=2, as_frame=True)
        df = raw.frame.copy()
        df.to_csv(cache_file, index=False)

    target = "class"
    df = df.dropna()
    if max_rows is not None:
        df = df.sample(n=min(max_rows, len(df)), random_state=seed)

    y = (df[target].astype(str).str.contains(">50K")).astype(int).to_numpy()
    X = df.drop(columns=[target])

    cat = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]

    try:  # sklearn >= 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    ct = ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", ohe, cat)]
    )
    Xenc = ct.fit_transform(X).astype(np.float32)
    names = list(num) + list(ct.named_transformers_["cat"].get_feature_names_out(cat))

    Xtr, Xte, ytr, yte = train_test_split(
        Xenc, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Dataset(Xtr, Xte, ytr, yte, [str(n) for n in names])
