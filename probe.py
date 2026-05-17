"""
probe.py — Hallucination probe classifier.

L2 Logistic Regression with out-of-fold threshold tuning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use predict / predict_proba instead.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        C_candidates = [0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]

        best_oof_acc = -1.0
        best_C = 0.01
        best_threshold = 0.5
        best_oof_probs = np.zeros(len(y))

        for C in C_candidates:
            oof_probs = np.zeros(len(y))

            for train_idx, val_idx in skf.split(X_scaled, y):
                clf = LogisticRegression(
                    C=C, penalty="l2", solver="lbfgs",
                    max_iter=5000, random_state=42,
                )
                clf.fit(X_scaled[train_idx], y[train_idx])
                oof_probs[val_idx] = clf.predict_proba(X_scaled[val_idx])[:, 1]

            for t in np.linspace(0.01, 0.99, 199):
                acc = accuracy_score(y, (oof_probs >= t).astype(int))
                if acc > best_oof_acc:
                    best_oof_acc = acc
                    best_C = C
                    best_threshold = float(t)
                    best_oof_probs = oof_probs.copy()

        self._threshold = best_threshold

        oof_auroc = roc_auc_score(y, best_oof_probs)
        print(f"  [OOF] Best C={best_C:.4f}  threshold={best_threshold:.3f}")
        print(f"  [OOF] Accuracy={best_oof_acc:.4f}  AUROC={oof_auroc:.4f}")

        self._clf = LogisticRegression(
            C=best_C, penalty="l2", solver="lbfgs",
            max_iter=5000, random_state=42,
        )
        self._clf.fit(X_scaled, y)

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._clf.predict_proba(X_scaled)
