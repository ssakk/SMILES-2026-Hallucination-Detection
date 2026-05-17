"""
splitting.py — 5-fold stratified cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []

    for idx_train_full, idx_test in skf.split(np.arange(len(y)), y):
        idx_train, idx_val = train_test_split(
            idx_train_full, test_size=0.15,
            stratify=y[idx_train_full], random_state=random_state,
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits
