
from __future__ import annotations
import time
import pandas as pd
from typing import Dict, List, Tuple
from .profiling import infer_profile
from .anomaly_methods import DETECTORS
from .anomaly_evaluate import inject_anomalies, f1_score_from_masks

def try_anomaly_methods(df: pd.DataFrame, profile, methods_map: Dict[str, List[str]], seeds: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cp in profile.columns:
        col = cp.name
        col_type = cp.dtype if cp.dtype in ("numeric", "categorical", "boolean", "datetime") else "numeric"
        methods = methods_map.get(col_type, list(DETECTORS.get(col_type, {}).keys()))
        results[col] = {}
        if df[col].dropna().shape[0] <= 5:
            continue
        for m in methods:
            det = DETECTORS.get(col_type, {}).get(m)
            if det is None: 
                continue
            metrics_accum = []
            t0 = time.time()
            for seed in seeds:
                corrupted, true_mask = inject_anomalies(df[col], col_type, frac=0.05, seed=seed)
                pred_mask = det(corrupted)
                metrics_accum.append(f1_score_from_masks(true_mask, pred_mask))
            avg = {
                "precision": float(sum(d["precision"] for d in metrics_accum)/len(metrics_accum)),
                "recall": float(sum(d["recall"] for d in metrics_accum)/len(metrics_accum)),
                "f1": float(sum(d["f1"] for d in metrics_accum)/len(metrics_accum)),
                "runtime_sec": float(time.time()-t0),
            }
            results[col][m] = avg
    return results

def pick_best_anomaly_per_column(results, dtype_map):
    selection = {}
    for col, m2m in results.items():
        if not m2m: 
            continue
        col_type = dtype_map.get(col, "numeric")
        best = max(m2m.items(), key=lambda kv: kv[1]["f1"])
        selection[col] = {"method": best[0], "metrics": best[1], "dtype": col_type}
    return selection

def apply_anomaly_treatment(df: pd.DataFrame, selection: Dict[str, dict]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Simple treatment: detected anomalies are set to NaN to be handled by imputation later.
    """
    from .anomaly_methods import DETECTORS
    df2 = df.copy()
    counts = {}
    for col, sel in selection.items():
        m = sel["method"]
        dtype = sel["dtype"]
        det = DETECTORS.get(dtype, {}).get(m)
        if det is None:
            continue
        mask = det(df2[col])
        counts[col] = int(mask.sum())
        df2.loc[mask, col] = pd.NA
    return df2, counts
