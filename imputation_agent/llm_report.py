
from __future__ import annotations
import json
from typing import Optional, Literal, Dict, Any
from .llm import load_llm

def generate_llm_json_report(profile: dict,
                             results: dict,
                             selection: dict,
                             provider: Literal["openai","ollama"] = "ollama",
                             model: Optional[str] = None,
                             temperature: float = 0.0) -> Dict[str, Any]:
    llm = load_llm(provider=provider, model=model, temperature=temperature)

    system = (
        "You are a data quality expert. Respond with STRICT JSON only.\n"
        "Use the provided metrics to justify choices. Include cautions about MNAR/leakage when relevant.\n"
    )
    user = {
        "task": "Generate imputation QA JSON report",
        "profile": profile,
        "results_per_method": results,
        "selection": selection,
        "constraints": {
            "json_only": True,
            "include_sections": [
                "dataset_summary",
                "column_decisions",
                "comparative_analysis",
                "global_recommendations",
                "risks_and_assumptions",
                "next_steps"
            ]
        }
    }

    prompt = f"""
{system}
User JSON:
{json.dumps(user)}

Return a SINGLE JSON object with keys: dataset_summary, column_decisions, comparative_analysis, global_recommendations, risks_and_assumptions, next_steps. No prose outside JSON.
"""
    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))
    try:
        text = text.replace("```json", "")
        text = text.replace("```", "")
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if m:
            return json.loads(m.group(0))
        raise
