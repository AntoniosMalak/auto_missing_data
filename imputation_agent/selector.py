
from __future__ import annotations

def pick_best_per_column(results, dtype_map):
    selection = {}
    for col, m2m in results.items():
        if not m2m: 
            continue
        col_type = dtype_map.get(col, "numeric")
        if col_type == "numeric":
            best = min(m2m.items(), key=lambda kv: kv[1].get("MAE", float("inf")))
        elif col_type in ("categorical","boolean"):
            best = max(m2m.items(), key=lambda kv: kv[1].get("ACC", -1.0))
        elif col_type == "datetime":
            best = min(m2m.items(), key=lambda kv: kv[1].get("MAE_days", float("inf")))
        else:
            # fallback: choose the one with minimal primary metric available
            def _score(v):
                for k in ("MAE","MAE_days"):
                    if k in v: return v[k]
                return 1e9
            best = min(m2m.items(), key=lambda kv: _score(kv[1]))
        selection[col] = {"method": best[0], "metrics": best[1], "dtype": col_type}
    return selection
