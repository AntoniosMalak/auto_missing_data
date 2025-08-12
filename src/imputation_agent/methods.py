
from __future__ import annotations
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def imputer_factory(method: str, col_type: str):
    if col_type == "numeric":
        if method == "mean": return SimpleImputer(strategy="mean")
        if method == "median": return SimpleImputer(strategy="median")
        if method == "knn": return KNNImputer(n_neighbors=5)
        if method == "mice": return IterativeImputer(random_state=0, sample_posterior=False, max_iter=10)
    if col_type == "categorical":
        if method == "most_frequent": return SimpleImputer(strategy="most_frequent")
        if method == "constant": return SimpleImputer(strategy="constant", fill_value="__MISSING__")
    if col_type == "boolean":
        return SimpleImputer(strategy="most_frequent")
    return None

def datetime_fill(series, method: str):
    import pandas as pd
    s = pd.to_datetime(series)
    if method == "ffill_bfill":
        return s.ffill().bfill()
    if method == "interpolate_linear":
        ts = s.view('int64')
        ts_interp = pd.Series(ts).interpolate(method="linear", limit_direction="both")
        return pd.to_datetime(ts_interp.astype('int64'))
    raise ValueError(f"Unknown datetime method: {method}")
