
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def mask_for_eval(df: pd.DataFrame, col: str, frac: float=0.1, seed: int=42):
    rng = np.random.default_rng(seed)
    not_missing = df[col].dropna().index
    k = max(1, int(len(not_missing) * frac))
    masked_idx = rng.choice(not_missing, size=k, replace=False)
    masked_df = df.copy()
    true_vals = masked_df.loc[masked_idx, col].copy()
    masked_df.loc[masked_idx, col] = np.nan
    return masked_df, masked_idx, true_vals

def score_numeric(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred))}

def average_metrics(list_of_dicts):
    keys = list(list_of_dicts[0].keys())
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}
