
from __future__ import annotations
import json
import pandas as pd
from typing import Optional
from .profiling import infer_profile, parse_datetimes_inplace
from .config import PipelineConfig
from .runner import run_pipeline
from .llm_report import generate_llm_json_report

def tool_profile(csv_path: str) -> str:
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
    cfg = cfg or PipelineConfig()
    out_csv, report_json, results, selection, profile = run_pipeline(csv_path, out_dir, cfg)
    return json.dumps({
        "imputed_csv": out_csv,
        "report_json": report_json,
        "all_methods_report": f"{out_dir}/all_methods_report.json",
        "selection": selection
    })

def tool_llm_report(profile: str, results: str, selection: str,
                    provider: str = "ollama", model: str | None = None, temperature: float = 0.0) -> str:
    profile = json.loads(profile)
    results = json.loads(results)
    selection = json.loads(selection)
    rep = generate_llm_json_report(profile, results, selection, provider=provider, model=model, temperature=temperature)
    return json.dumps(rep, ensure_ascii=False)
