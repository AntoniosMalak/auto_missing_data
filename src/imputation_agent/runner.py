
from __future__ import annotations
import os, json, time
import pandas as pd
import numpy as np
from typing import Dict, List
from joblib import dump
from .config import PipelineConfig
from .profiling import infer_profile, parse_datetimes_inplace
from .methods import imputer_factory
from .evaluate import mask_for_eval, score_numeric, average_metrics
from .selector import pick_best_per_column


def try_methods(df: pd.DataFrame, profile, methods_map: Dict[str, List[str]], seeds: List[int]):
    results = {}
    for cp in profile.columns:
        if cp.missing_rate == 0:
            continue
        col = cp.name
        col_methods = methods_map.get(cp.dtype, [])
        results[col] = {}
        for m in col_methods:
            metrics_list = []
            for seed in seeds:
                masked_df, idx, true_vals = mask_for_eval(df, col, frac=0.1, seed=seed)
                if cp.dtype == "datetime":
                    pred = pd.to_datetime(masked_df[col]).ffill().bfill().loc[idx]
                    tv = pd.to_datetime(true_vals)
                    pv = pd.to_datetime(pred)
                    mae_days = np.mean(np.abs((tv - pv).dt.total_seconds())/86400.0)
                    metrics = {"MAE_days": float(mae_days)}
                elif cp.dtype in ("categorical","boolean"):
                    imp = imputer_factory(m, cp.dtype)
                    X = masked_df[[col]]
                    yhat = pd.Series(imp.fit_transform(X).ravel(), index=X.index).loc[idx]
                    acc = float((yhat == true_vals).mean())
                    metrics = {"ACC": acc}
                else:  # numeric + unknown fallback
                    imp = imputer_factory(m, "numeric")
                    X = masked_df[[col]]
                    yhat = pd.Series(imp.fit_transform(X).ravel(), index=X.index).loc[idx]
                    metrics = score_numeric(true_vals, yhat)
                metrics_list.append(metrics)
            results[col][m] = average_metrics(metrics_list)
    return results


def impute_full(df: pd.DataFrame, selection: Dict[str,Dict]):
    df_imp = df.copy()
    # group by (dtype, method)
    groups = {}
    for col, info in selection.items():
        key = (info["dtype"], info["method"])
        groups.setdefault(key, []).append(col)

    imputers = {}
    for (dtype, method), cols in groups.items():
        if dtype == "datetime":
            for c in cols:
                df_imp[c] = pd.to_datetime(df_imp[c]).ffill().bfill()
            continue
        imp = imputer_factory(method, dtype if dtype in ("numeric","categorical","boolean") else "numeric")
        X = df_imp[cols]
        df_imp[cols] = imp.fit_transform(X)
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

    out_csv = os.path.join(out_dir, "imputed.csv")
    df_imp.to_csv(out_csv, index=False)
    dump(imputers, os.path.join(out_dir, "imputers.joblib"))

    report = {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "selection": selection,
        "methods_tried": methods_map,
        "timestamp": time.time(),
    }
    report_json = os.path.join(out_dir, "imputation_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return out_csv, report_json, results, selection, profile
