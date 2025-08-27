
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict

def inject_anomalies(series: pd.Series, dtype: str, frac: float = 0.05, seed: int = 42) -> Tuple[pd.Series, pd.Series]:
    """Returns (corrupted_series, is_anomaly_mask)"""
    rng = np.random.default_rng(seed)
    s = series.copy()

    # candidates = currently non-missing
    candidates = s.index[s.notna()]
    n = len(candidates)
    if n == 0:
        return s, pd.Series(False, index=s.index)

    k = max(1, int(n * frac))
    idx = pd.Index(rng.choice(candidates, size=k, replace=False))

    mask = pd.Series(False, index=s.index)

    if dtype == "numeric":
        # If it's an integer dtype, upcast to float64 to allow float spikes safely
        if pd.api.types.is_integer_dtype(s.dtype):
            s = s.astype("float64")
        # Generate spikes far from mean
        mu = pd.to_numeric(s, errors="coerce").mean()
        sigma = pd.to_numeric(s, errors="coerce").std(ddof=0) or 1.0
        spikes = mu + (5 + rng.normal(0, 1, size=k)) * sigma  # big deviations
        s.loc[idx] = spikes
        mask.loc[idx] = True

    elif dtype == "datetime":
        # Coerce to datetime, preserving tz if present
        dt = pd.to_datetime(s, errors="coerce")
        # Build ±(~5 years) deltas in days; preserves tz automatically
        deltas = pd.to_timedelta(rng.choice([-1, 1], size=k) * 365 * 5, unit="D")
        # Assign shifted timestamps at the chosen indices
        shifted = dt.loc[idx] + deltas
        dt.loc[idx] = shifted
        s = dt
        mask.loc[idx] = True

    elif dtype in ("categorical", "boolean"):
        # Switch to object so we can inject unseen tokens without dtype complaints
        s = s.astype("object")
        new_vals = [f"__UNSEEN_{i}__" for i in range(k)]
        s.loc[idx] = new_vals
        mask.loc[idx] = True

    else:
        # Fallback: do nothing, but return the mask (all False)
        pass

    return s, mask

def f1_score_from_masks(true_mask: pd.Series, pred_mask: pd.Series) -> Dict[str, float]:
    tp = int(((true_mask) & (pred_mask)).sum())
    fp = int((~true_mask & pred_mask).sum())
    fn = int((true_mask & ~pred_mask).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
