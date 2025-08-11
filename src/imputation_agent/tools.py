
from __future__ import annotations
import json
from typing import Optional
import pandas as pd
from .profiling import infer_profile, parse_datetimes_inplace
from .config import PipelineConfig
from .runner import run_pipeline


def tool_profile(csv_path: str) -> str:
    """Profile the CSV and return a JSON summary string."""
    df = pd.read_csv(csv_path)
    parse_datetimes_inplace(df)
    prof = infer_profile(df)
    return json.dumps({
        "n_rows": prof.n_rows,
        "n_cols": prof.n_cols,
        "columns": [vars(c) for c in prof.columns],
        "has_datetime_index": prof.has_datetime_index
    })


def tool_run_pipeline(csv_path: str, out_dir: str = "outputs", cfg: Optional[PipelineConfig] = None) -> str:
    """Run the full pipeline and return output paths as JSON string."""
    cfg = cfg or PipelineConfig()
    out_csv, report_json, results, selection, profile = run_pipeline(csv_path, out_dir, cfg)
    return json.dumps({
        "imputed_csv": out_csv,
        "report_json": report_json,
        "selection_columns": list(selection.keys())
    })
