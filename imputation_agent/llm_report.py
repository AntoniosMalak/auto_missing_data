
from __future__ import annotations
import json, re
from typing import Optional, Literal, Dict, Any
from .llm import load_llm

def _coerce_json(text: str) -> Dict[str, Any]:
    # strip code fences and try to load
    t = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", t)  
        if match:
            return json.loads(match.group(0))
        raise

def generate_llm_json_report(profile: dict,
                             results: dict,
                             selection: dict,
                             provider: Literal["openai","ollama"] = "ollama",
                             model: Optional[str] = None,
                             temperature: float = 0.0) -> Dict[str, Any]:
    llm = load_llm(provider=provider, model=model, temperature=temperature)

    system = (
        "You are a data quality expert. Respond with STRICT JSON only. "
        "Use the provided metrics to justify choices. Include cautions about MNAR/leakage when relevant. "
        "If anomaly info is present (keys 'anomaly_results' or 'anomaly_selection'), include it."
    )
    user = {
        "task": "Generate comprehensive imputation + anomaly QA JSON report",
        "profile": profile,
        "results_per_method": results,
        "selection": selection,
        # Optional anomaly keys if caller included them inside results
        "anomaly_results": results.get("anomaly_results", {}),
        "anomaly_selection": results.get("anomaly_selection", {}),
        "constraints": {
            "json_only": True,
            "include_sections": [
                "dataset_summary",
                "anomaly_decisions",
                "imputation_decisions",
                "comparative_analysis",
                "global_recommendations",
                "risks_and_assumptions",
                "next_steps",
                "artifacts"
            ]
        }
    }

    prompt = f"""{system}

User JSON:
{json.dumps(user)}

Return a SINGLE JSON object with exactly these keys:
- dataset_summary
- anomaly_decisions
- imputation_decisions
- comparative_analysis
- global_recommendations
- risks_and_assumptions
- next_steps
- artifacts

Where:
* anomaly_decisions: per-column detector chosen (if any) with F1/precision/recall and rationale.
* imputation_decisions: per-column imputer chosen with metrics and rationale (e.g., why mean over median, KNN limits, dtype handling).
* comparative_analysis: contrasts for top-2 methods per column where close.
* artifacts: absolute/relative paths of produced files (imputed csv, reports) if provided in results.
No prose outside JSON.
"""
    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))
    data = _coerce_json(text)

    # ensure minimally required keys exist
    for k in ["dataset_summary","anomaly_decisions","imputation_decisions",
              "comparative_analysis","global_recommendations","risks_and_assumptions",
              "next_steps","artifacts"]:
        data.setdefault(k, {} if k not in ("global_recommendations","risks_and_assumptions","next_steps") else [])

    return data
