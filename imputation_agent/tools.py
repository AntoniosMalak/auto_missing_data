
from __future__ import annotations
import json
import pandas as pd
from typing import Optional, Any, Dict
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
    }, ensure_ascii=False)

def tool_run_pipeline(csv_path: str, out_dir: str = "outputs", cfg: Optional[PipelineConfig] = None) -> str:
    cfg = cfg or PipelineConfig()
    out_csv, report_json, results, selection, profile = run_pipeline(csv_path, out_dir, cfg)
    return json.dumps({
        "imputed_csv": out_csv,
        "report_json": report_json,
        "all_methods_report": f"{out_dir}/all_methods_report.json",
        "selection": selection
    }, ensure_ascii=False)

def _ensure_obj(x: Any) -> Dict:
    if isinstance(x, str):
        return json.loads(x)
    if isinstance(x, dict):
        return x
    raise ValueError("tool_llm_report expects dict or JSON string inputs")

def tool_llm_report(profile: str | dict, results: str | dict, selection: str | dict,
                    provider: str = "ollama", model: str | None = None, temperature: float = 0.0) -> str:
    profile_obj = _ensure_obj(profile)
    results_obj = _ensure_obj(results)
    selection_obj = _ensure_obj(selection)
    # Merge artifact hints if present in results_obj (to help LLM populate artifacts)
    if "artifacts" not in results_obj and isinstance(selection_obj, dict):
        pass
    rep = generate_llm_json_report(profile_obj, results_obj, selection_obj,
                                   provider=provider, model=model, temperature=temperature)
    return json.dumps(rep, ensure_ascii=False)
