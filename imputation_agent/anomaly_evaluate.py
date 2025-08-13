
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict

def inject_anomalies(series: pd.Series, dtype: str, frac: float = 0.05, seed: int = 42) -> Tuple[pd.Series, pd.Series]:
    """Returns (corrupted_series, is_anomaly_mask)"""
    rng = np.random.default_rng(seed)
    s = series.copy()
    n = len(s.dropna())
    k = max(1, int(n * frac))
    idx = rng.choice(s.dropna().index, size=k, replace=False)
    mask = pd.Series(False, index=s.index)

    if dtype == "numeric":
        mu = s.mean()
        sigma = s.std(ddof=0) or 1.0
        spikes = mu + (5 + rng.normal(0, 1, size=k)) * sigma
        s.loc[idx] = spikes
        mask.loc[idx] = True
    elif dtype == "datetime":
        dt = pd.to_datetime(s, errors="coerce")
        epoch = dt.view("int64")
        shift = int(60*60*24*365 * 1e9)  # ~1 year in ns
        anomalies = pd.to_datetime(epoch + (rng.choice([-1,1], size=k) * 5 * shift))
        s.loc[idx] = anomalies
        mask.loc[idx] = True
    elif dtype in ("categorical","boolean"):
        s = s.astype("object")
        new_vals = [f"__UNSEEN_{i}__" for i in range(k)]
        s.loc[idx] = new_vals
        mask.loc[idx] = True
    return s, mask

def f1_score_from_masks(true_mask: pd.Series, pred_mask: pd.Series) -> Dict[str, float]:
    tp = int(((true_mask) & (pred_mask)).sum())
    fp = int((~true_mask & pred_mask).sum())
    fn = int((true_mask & ~pred_mask).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
