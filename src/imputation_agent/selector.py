
from __future__ import annotations
from typing import Dict


def pick_best_per_column(results: Dict[str, Dict[str, Dict]], dtype_map: Dict[str,str]):
    selection = {}
    for col, m2m in results.items():
        if not m2m:
            continue
        col_type = dtype_map.get(col, "numeric")
        if col_type == "numeric":
            best = min(m2m.items(), key=lambda kv: kv[1]["MAE"])
        elif col_type in ("categorical","boolean"):
            best = max(m2m.items(), key=lambda kv: kv[1]["ACC"])
        elif col_type == "datetime":
            best = min(m2m.items(), key=lambda kv: kv[1]["MAE_days"])
        else:
            # default to lowest MAE if unknown
            # (assumes numeric-like)
            best = min(m2m.items(), key=lambda kv: list(kv[1].values())[0])
        selection[col] = {"method": best[0], "metrics": best[1], "dtype": col_type}
    return selection
