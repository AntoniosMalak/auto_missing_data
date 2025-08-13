
from __future__ import annotations
import os, json, time, warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from joblib import dump
from .config import PipelineConfig
from .profiling import infer_profile, parse_datetimes_inplace
from .methods import imputer_factory, datetime_fill
from .evaluate import mask_for_eval, score_numeric, average_metrics
from .selector import pick_best_per_column

def _cast_for_imputer(X: pd.DataFrame, col_type: str) -> pd.DataFrame:
    """Sklearn SimpleImputer refuses dtype=bool. For categorical/boolean, cast to object."""
    if col_type in ("categorical", "boolean"):
        return X.astype("object")
    return X

def _restore_boolean_dtype(s: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    def coerce(v):
        if isinstance(v, bool): return v
        if isinstance(v, (int, np.integer)): return bool(v)
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv in mapping: return mapping[lv]
        return v
    out = s.map(coerce)
    if out.dropna().map(lambda x: isinstance(x, bool)).all():
        return out.astype(bool)
    return out

def _safe_apply(method_name: str, col: str, col_type: str, masked_df: pd.DataFrame, true_vals: pd.Series, idx) -> Dict[str, float]:
    try:
        if col_type == "datetime":
            chosen = method_name if method_name in ("ffill_bfill","interpolate_linear") else "ffill_bfill"
            pred_full = datetime_fill(masked_df[col], chosen)
            pred = pred_full.loc[idx]
            tv = pd.to_datetime(true_vals)
            pv = pd.to_datetime(pred)
            mae_days = float(np.mean(np.abs((tv - pv).dt.total_seconds())/86400.0))
            return {"MAE_days": mae_days}
        elif col_type in ("categorical","boolean"):
            imp = imputer_factory(method_name, col_type)
            X = _cast_for_imputer(masked_df[[col]], col_type)
            yhat = pd.Series(imp.fit_transform(X).ravel(), index=X.index).loc[idx]
            y_true = true_vals
            try:
                yhat_cmp = _restore_boolean_dtype(yhat) if col_type == "boolean" else yhat
                ytrue_cmp = _restore_boolean_dtype(y_true) if col_type == "boolean" else y_true
            except Exception:
                yhat_cmp, ytrue_cmp = yhat, y_true
            acc = float((pd.Series(yhat_cmp).reset_index(drop=True) == pd.Series(ytrue_cmp).reset_index(drop=True)).mean())
            return {"ACC": acc}
        else:
            imp = imputer_factory(method_name, "numeric")
            X = masked_df[[col]]
            yhat = pd.Series(imp.fit_transform(X).ravel(), index=X.index).loc[idx]
            return score_numeric(true_vals, yhat)
    except Exception as e:
        warnings.warn(f"[{col}] method={method_name} failed: {e}")
        if col_type in ("categorical","boolean"): return {"ACC": -1.0}
        if col_type == "datetime": return {"MAE_days": float("inf")}
        return {"MAE": float("inf")}

def try_methods(df: pd.DataFrame, profile, methods_map: Dict[str, List[str]], seeds: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cp in profile.columns:
        col = cp.name
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            results[col] = {}
            continue
        col_type = cp.dtype if cp.dtype in ("numeric", "categorical", "boolean", "datetime") else "numeric"
        methods = methods_map.get(col_type, methods_map.get("numeric", []))
        results[col] = {}
        if df[col].dropna().shape[0] <= 1:
            continue
        for m in methods:
            metrics_list = []
            t0 = time.time()
            for seed in seeds:
                masked_df, idx, true_vals = mask_for_eval(df, col, frac=0.1, seed=seed)
                metrics = _safe_apply(m, col, col_type, masked_df, true_vals, idx)
                metrics_list.append(metrics)
            avg = average_metrics(metrics_list)
            avg["runtime_sec"] = float(time.time() - t0)
            results[col][m] = avg
    return results

def impute_full(df: pd.DataFrame, selection: Dict[str,Dict]) -> Tuple[pd.DataFrame, Dict[Tuple[str,str], object]]:
    df_imp = df.copy()
    imputers: Dict[Tuple[str,str], object] = {}
    buckets: Dict[Tuple[str,str], List[str]] = {}
    for col, info in selection.items():
        key = (info["dtype"], info["method"])
        buckets.setdefault(key, []).append(col)
    for (dtype, method), cols in buckets.items():
        if dtype == "datetime":
            chosen = method if method in ("ffill_bfill","interpolate_linear") else "ffill_bfill"
            for c in cols:
                df_imp[c] = datetime_fill(df_imp[c], chosen)
            continue
        imp = imputer_factory(method, dtype if dtype in ("numeric","categorical","boolean") else "numeric")
        X = _cast_for_imputer(df_imp[cols], dtype)
        transformed = imp.fit_transform(X)
        df_imp[cols] = transformed
        if dtype == "boolean":
            for c in cols:
                df_imp[c] = _restore_boolean_dtype(df_imp[c])
        imputers[(dtype, method)] = imp
    return df_imp, imputers

def run_pipeline(csv_path: str, out_dir: str, cfg: PipelineConfig):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    parse_datetimes_inplace(df)
    profile = infer_profile(df)
    dtype_map = {cp.name: cp.dtype for cp in profile.columns}

    methods_map = {
        "numeric": cfg.methods.numeric,
        "categorical": cfg.methods.categorical,
        "boolean": cfg.methods.boolean,
        "datetime": cfg.methods.datetime,
    }
    results = try_methods(df, profile, methods_map, seeds=cfg.evaluation.seeds)
    selection = pick_best_per_column(results, dtype_map)
    df_imp, imputers = impute_full(df, selection)

    all_methods_path = os.path.join(out_dir, "all_methods_report.json")
    with open(all_methods_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(out_dir, "imputed.csv")
    df_imp.to_csv(out_csv, index=False)
    dump(imputers, os.path.join(out_dir, "imputers.joblib"))

    report = {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "selection": selection,
        "timestamp": time.time(),
        # helpful artifact paths for LLM report
        "artifacts": {
            "imputed_csv": out_csv,
            "all_methods_report": all_methods_path,
            "report_json": os.path.join(out_dir, "imputation_report.json"),
        },
    }
    report_json = os.path.join(out_dir, "imputation_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_csv, report_json, results, selection, profile
