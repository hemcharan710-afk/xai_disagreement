"""
Differentiable models for the Krishna et al. reproduction.

The paper applies gradient explanation methods to the logistic regression and
neural-network models (gradients are undefined for the tree models), so both are
implemented in torch here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TorchLogReg(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)

    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return torch.softmax(self(torch.tensor(np.asarray(X, np.float32))), 1).numpy()


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(50, 100, 50), n_classes: int = 2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return torch.softmax(self(torch.tensor(np.asarray(X, np.float32))), 1).numpy()


def train(model: nn.Module, ds, epochs: int = 60, lr: float = 1e-3,
          weight_decay: float = 1e-4, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(ds.X_train, dtype=torch.float32)
    yt = torch.tensor(ds.y_train, dtype=torch.long)
    bs = 256
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xt[b]), yt[b]).backward()
            opt.step()
    return model


def accuracy(model, X, y) -> float:
    return float((model.predict_proba(X).argmax(1) == y).mean())
