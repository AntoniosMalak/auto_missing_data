from __future__ import annotations
import json, re
from typing import Optional, Literal, Dict, Any
from .llm import load_llm

def _coerce_json(text: str) -> Dict[str, Any]:
    # Remove code fences and comments
    t = text.replace("```json", "").replace("```", "").strip()
    t = re.sub(r'^\s*//.*$', '', t, flags=re.MULTILINE)
    # Extract the first JSON object in the string
    match = re.search(r"\{(?:[^{}]|(?R))*\}", t, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}\nExtracted: {match.group(0)}")
    raise ValueError("No JSON object found in LLM response")

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
* anomaly_decisions: For each column, detail the chosen detector. The explanation for each MUST include:
    1. **Reasoning**: A clear justification for the choice based on its F1-score compared to other methods.
    2. **Advantage**: The main strength of the chosen method for this specific column's data type (e.g., 'Z-score is simple and effective for normally distributed data').
    3. **Weakness**: A potential drawback or assumption of the method (e.g., 'Z-score is sensitive to extreme outliers, which can inflate the standard deviation').
* imputation_decisions: For each column, detail the chosen imputer. The explanation for each MUST include:
    1. **Reasoning**: A clear justification for the choice based on its performance metric (e.g., lowest MAE for numeric, highest ACC for categorical).
    2. **Advantage**: The primary benefit of the method (e.g., 'MICE can capture complex relationships between variables, potentially leading to more accurate imputations').
    3. **Weakness**: A limitation or key assumption (e.g., 'MICE assumes the data is Missing At Random (MAR) and is computationally more intensive than simple methods').
* comparative_analysis: Contrasts the top-2 methods per column where performance was close, explaining the trade-offs.
* artifacts: Absolute/relative paths of produced files (imputed csv, reports) if provided in results.
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