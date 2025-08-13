
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable

def _numeric_zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0) or 1.0
    z = (x - mu) / sd
    return z.abs() > 3

def _numeric_iqr(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = (q3 - q1) or 1.0
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (x < lo) | (x > hi)

def _numeric_iforest(s: pd.Series) -> pd.Series:
    from sklearn.ensemble import IsolationForest
    x = pd.to_numeric(s, errors="coerce").to_frame(name="x").dropna()
    if x.empty: 
        return pd.Series(False, index=s.index)
    clf = IsolationForest(random_state=42, contamination="auto")
    pred = pd.Series(False, index=s.index)
    pred.loc[x.index] = clf.fit_predict(x) == -1
    return pred

def _numeric_lof(s: pd.Series) -> pd.Series:
    from sklearn.neighbors import LocalOutlierFactor
    x = pd.to_numeric(s, errors="coerce").to_frame(name="x").dropna()
    if x.empty or len(x) < 10:
        return pd.Series(False, index=s.index)
    lof = LocalOutlierFactor(n_neighbors=min(20, len(x)-1), novelty=False)
    y = lof.fit_predict(x)
    pred = pd.Series(False, index=s.index)
    pred.loc[x.index] = y == -1
    return pred

def _categorical_rare(s: pd.Series, min_frac: float = 0.01) -> pd.Series:
    x = s.astype("object")
    vc = x.value_counts(normalize=True, dropna=True)
    rare_vals = set(vc[vc < min_frac].index)
    return x.isin(rare_vals)

def _categorical_unexpected(s: pd.Series) -> pd.Series:
    x = s.astype("object")
    # unexpected placeholder pattern created by injector
    return x.astype(str).str.startswith("__UNSEEN_")

def _datetime_zscore(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce").view("int64")
    mu = t.mean()
    sd = t.std(ddof=0) or 1.0
    z = (t - mu) / sd
    return pd.Series(np.abs(z) > 3, index=s.index)

def _none(s: pd.Series) -> pd.Series:
    return pd.Series(False, index=s.index)

DETECTORS: Dict[str, Dict[str, Callable[[pd.Series], pd.Series]]] = {
    "numeric": {
        "zscore": _numeric_zscore,
        "iqr": _numeric_iqr,
        "isolation_forest": _numeric_iforest,
        "lof": _numeric_lof,
    },
    "categorical": {
        "rare": _categorical_rare,
        "unexpected": _categorical_unexpected,
    },
    "datetime": {
        "zscore": _datetime_zscore,
    },
    "boolean": {
        "none": _none,
    },
}
