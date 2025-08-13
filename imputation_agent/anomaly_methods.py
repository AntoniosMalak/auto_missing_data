
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ---------- Numeric detectors ----------
def zscore_numeric(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m = x.mean()
    sd = x.std(ddof=0)
    z = (x - m) / (sd if sd else 1.0)
    return z.abs() > threshold

def iqr_numeric(s: pd.Series, factor: float = 1.5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    return (x < lo) | (x > hi)

def iso_forest_numeric(s: pd.Series, contamination: float = 0.05, random_state: int = 0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").values.reshape(-1, 1)
    mask = ~np.isnan(x).ravel()
    labels = np.zeros(len(s), dtype=bool)
    if mask.sum() > 1:
        clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
        preds = clf.fit_predict(x[mask])
        labels[mask] = preds == -1
    return pd.Series(labels, index=s.index)

def lof_numeric(s: pd.Series, contamination: float = 0.05) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").values.reshape(-1, 1)
    mask = ~np.isnan(x).ravel()
    labels = np.zeros(len(s), dtype=bool)
    if mask.sum() > 10:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        preds = lof.fit_predict(x[mask])
        labels[mask] = preds == -1
    return pd.Series(labels, index=s.index)

# ---------- Datetime detectors ----------
def zscore_datetime(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    epoch = dt.view("int64")
    m = epoch.mean()
    sd = epoch.std(ddof=0)
    z = (epoch - m) / (sd if sd else 1.0)
    return z.abs() > threshold

# ---------- Categorical detectors ----------
def rare_categories(s: pd.Series, min_frac: float = 0.01) -> pd.Series:
    vc = s.value_counts(dropna=True, normalize=True)
    rare = set(vc[vc < min_frac].index)
    return s.isin(rare)

def unexpected_values(s: pd.Series, allowed: set | None = None) -> pd.Series:
    if allowed is None:
        allowed = set(s.dropna().unique())
    return ~s.isna() & ~s.isin(list(allowed))

# ---------- Router ----------
DETECTORS = {
    "numeric": {
        "zscore": zscore_numeric,
        "iqr": iqr_numeric,
        "isolation_forest": iso_forest_numeric,
        "lof": lof_numeric,
    },
    "datetime": {
        "zscore": zscore_datetime,
    },
    "categorical": {
        "rare": rare_categories,
        "unexpected": unexpected_values,
    },
    "boolean": {
        # Booleans rarely have "anomalies"; return all False
        "none": lambda s: pd.Series(False, index=s.index),
    },
}
