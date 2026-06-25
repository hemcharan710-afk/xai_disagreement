"""
Basic CNN and the same four-model degradation spectrum, for MNIST.

The CNN accepts flat 784-vectors (reshaping to 1x28x28 internally) so every
flat-vector explainer in ``xai_disagreement.explainers`` works without change.

    A  Control       clean data, weight decay                  (High Gen.)
    C  Feature Noise Gaussian pixel noise during training      (Low Gen.)
    D  Label Poison  20% of training labels randomized         (Low Gen.)
    B  Overfit       large head, no reg, memorizes a subset    (Worst Gen.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

SPECTRUM = ["A", "C", "D", "B"]
VARIANT_LABEL = {
    "A": "A - Control",
    "C": "C - Feat. Noise",
    "D": "D - Label Poison",
    "B": "B - Overfit",
}


class BasicCNN(nn.Module):
    def __init__(self, n_classes: int = 10, head: int = 128, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, head), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head, n_classes),
        )

    def forward(self, x):
        if x.dim() == 2:                 # flat (N, 784) -> (N, 1, 28, 28)
            x = x.view(-1, 1, 28, 28)
        return self.classifier(self.features(x))

    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(X, dtype=np.float32))
            return torch.softmax(self(xt), dim=1).numpy()


@dataclass
class TrainedModel:
    variant: str
    model: BasicCNN
    train_acc: float
    test_acc: float
    gen_gap: float = field(init=False)

    def __post_init__(self):
        self.gen_gap = self.train_acc - self.test_acc


def _acc(model, X, y, bs=2000) -> float:
    preds = np.concatenate([model.predict_proba(X[i:i + bs]).argmax(1)
                            for i in range(0, len(X), bs)])
    return float((preds == y).mean())


def train_variant(variant: str, ds, seed: int = 0, epochs: int | None = None
                  ) -> TrainedModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr = ds.X_train.copy(), ds.y_train.copy()

    if variant == "A":
        head, drop, wd, ep, sub, noise = 128, 0.25, 1e-3, 6, None, 0.0
    elif variant == "C":
        head, drop, wd, ep, sub, noise = 128, 0.25, 1e-3, 6, None, 0.6
    elif variant == "D":
        head, drop, wd, ep, sub, noise = 128, 0.25, 1e-3, 6, None, 0.0
        flip = rng.random(len(ytr)) < 0.20
        ytr[flip] = rng.integers(0, ds.n_classes, size=int(flip.sum()))
    elif variant == "B":
        head, drop, wd, ep, sub, noise = 512, 0.0, 0.0, 60, 1500, 0.0
    else:
        raise ValueError(variant)
    if epochs is not None:
        ep = epochs

    if sub is not None:
        idx = rng.choice(len(Xtr), size=min(sub, len(Xtr)), replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    model = BasicCNN(ds.n_classes, head=head, dropout=drop)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    bs = 128
    model.train()
    for _ in range(ep):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            xb = Xt[b]
            if noise > 0:
                xb = torch.clamp(xb + noise * torch.randn_like(xb), 0, 1)
            opt.zero_grad()
            lossf(model(xb), yt[b]).backward()
            opt.step()

    return TrainedModel(variant, model, _acc(model, ds.X_train, ds.y_train),
                        _acc(model, ds.X_test, ds.y_test))
