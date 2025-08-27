from __future__ import annotations
import json, re
from typing import Optional, Literal, Dict, Any
from .llm import load_llm


IMPUTATION_KB: Dict[str, Dict[str, Any]] = {
    "numeric": {
        "mean": {
            "advantages": [
                "Fast and deterministic",
                "Works well when distribution is symmetric and unimodal",
            ],
            "weaknesses": [
                "Biased under skew/heavy tails",
                "Shrinks variance; can distort correlations",
                "Unsafe if missingness is MNAR",
            ],
            "assumptions": [
                "MCAR/MAR missingness preferred over MNAR",
                "Roughly symmetric distribution",
            ],
            "failure_modes": [
                "Right/left-skewed data (mean dragged by outliers)",
                "When column is a proxy for a sensitive label → leakage risk",
            ],
            "guardrails": [
                "If |skewness|>1 or high IQR/SD → prefer median or model-based",
            ]
        },
        "median": {
            "advantages": [
                "Robust to outliers and skew",
                "Preserves central tendency better on heavy-tailed data",
            ],
            "weaknesses": [
                "Ignores correlation with other features",
                "Can reduce variance in small samples",
            ],
            "assumptions": [
                "MCAR/MAR preferred; resilient under skew",
            ],
            "failure_modes": [
                "When strong multivariate structure is present (KNN/MICE better)",
            ],
            "guardrails": [
                "Switch to KNN/MICE when missing_rate is moderate and sample is large enough",
            ]
        },
        "knn": {
            "advantages": [
                "Uses multivariate similarity; can capture non-linear structure",
            ],
            "weaknesses": [
                "Computationally heavier; sensitive to scaling and choice of k",
                "Degrades when missing_rate is high or data is sparse",
            ],
            "assumptions": [
                "Locality assumption: similar rows have similar values",
                "Features reasonably scaled; no extreme sparsity",
            ],
            "failure_modes": [
                "High-dimensional data without scaling",
                "MNAR: neighbors don’t fix selection bias",
            ],
            "guardrails": [
                "Cap missing_rate (e.g., <= 0.4 per your config)",
                "Standardize/scale numeric features",
            ]
        },
        "mice": {
            "advantages": [
                "Model-based; leverages relationships across many columns",
                "Often best accuracy when assumptions hold",
            ],
            "weaknesses": [
                "Slow on large data; can be unstable with small n or collinearity",
                "Risk of overfitting or leakage if target info leaks into predictors",
            ],
            "assumptions": [
                "MAR preferred; models correctly specified",
            ],
            "failure_modes": [
                "Too many features vs rows; convergence issues",
                "MNAR patterns unaddressed",
            ],
            "guardrails": [
                "Limit rows/features; set random_state; review model diagnostics",
            ]
        },
    },
    "categorical": {
        "most_frequent": {
            "advantages": ["Simple, stable, preserves valid categories"],
            "weaknesses": ["Inflates the mode; distorts class balance"],
            "assumptions": ["MCAR/MAR; mode is representative"],
            "failure_modes": ["Rare classes get suppressed"],
            "guardrails": ["Consider 'constant' token if distribution is flat or noisy"]
        },
        "constant": {
            "advantages": ["Adds explicit '__MISSING__' signal", "Avoids pretending a class"],
            "weaknesses": ["Creates artificial class; downstream models must handle it"],
            "assumptions": ["Downstream accepts sentinel categories"],
            "failure_modes": ["Can inflate model capacity via new category"],
            "guardrails": ["Document sentinel; consider target encoding/OOH handling"]
        },
    },
    "boolean": {
        "most_frequent": {
            "advantages": ["Stable, simple"],
            "weaknesses": ["Biases toward majority class"],
            "assumptions": ["MCAR/MAR"],
            "failure_modes": ["When class imbalance is severe or correlated with target"],
            "guardrails": ["Audit class balance before/after imputation"]
        }
    },
    "datetime": {
        "ffill_bfill": {
            "advantages": ["Good for time-ordered continuity; simple and fast"],
            "weaknesses": ["Smears periods; can propagate stale values"],
            "assumptions": ["Temporal continuity; minimal regime shifts"],
            "failure_modes": ["Irregular sampling; structural breaks"],
            "guardrails": ["Prefer within-entity sorting; combine with anomaly removal first"]
        },
        "interpolate_linear": {
            "advantages": ["Captures linear trend between known timestamps"],
            "weaknesses": ["Inaccurate with seasonality or non-linear dynamics"],
            "assumptions": ["Rough linearity between known points"],
            "failure_modes": ["Strong seasonality/holiday effects"],
            "guardrails": ["Post-check with residual diagnostics"]
        },
    },
}

ANOMALY_KB: Dict[str, Dict[str, Any]] = {
    "numeric": {
        "zscore": {
            "advantages": ["Fast; well-understood; OK for near-Gaussian data"],
            "weaknesses": ["Breaks under heavy skew/outliers (SD inflated)"],
            "assumptions": ["Approximately normal distribution"],
            "failure_modes": ["Fat tails; multi-modal distributions"],
            "guardrails": ["Use robust (IQR) when skewed"]
        },
        "iqr": {
            "advantages": ["Robust to outliers; non-parametric"],
            "weaknesses": ["Threshold (1.5×IQR) arbitrary; misses global shifts"],
            "assumptions": ["Quartiles meaningful"],
            "failure_modes": ["Strong multimodality; small N"],
            "guardrails": ["Tune multiplier; combine with context checks"]
        },
        "isolation_forest": {
            "advantages": ["Handles non-linear structure; scales reasonably"],
            "weaknesses": ["Parameter sensitivity; stochastic"],
            "assumptions": ["Isolation path length reflects rarity"],
            "failure_modes": ["Tiny datasets; inappropriate contamination"],
            "guardrails": ["Fix random_state; review contamination inference"]
        },
        "lof": {
            "advantages": ["Local density aware; good for clusters"],
            "weaknesses": ["Needs enough neighbors; sensitive to k"],
            "assumptions": ["Density contrast reveals outliers"],
            "failure_modes": ["Small N; uneven sampling"],
            "guardrails": ["Ensure N>k*~1.5; scale features"]
        },
    },
    "categorical": {
        "rare": {
            "advantages": ["Flags low-frequency categories"],
            "weaknesses": ["Drifts with window; brittle with small N"],
            "assumptions": ["Rarity ≈ anomaly"],
            "failure_modes": ["Legit emergent categories"],
            "guardrails": ["Tune min_frac; whitelist known rare codes"]
        },
        "unexpected": {
            "advantages": ["Catches unseen tokens (e.g., __UNSEEN_*)"],
            "weaknesses": ["Relies on placeholder convention"],
            "assumptions": ["Unexpected tokens encode anomalies"],
            "failure_modes": ["Data without such placeholders"],
            "guardrails": ["Pair with 'rare' for production data"]
        },
    },
    "datetime": {
        "zscore": {
            "advantages": ["Simple detector on timestamp scale"],
            "weaknesses": ["Meaningless if cadence irregular"],
            "assumptions": ["Comparable epoch scale; few regime shifts"],
            "failure_modes": ["Time zone jumps; DST; reindexing errors"],
            "guardrails": ["Validate monotonicity; check time zone handling"]
        },
    },
    "boolean": {
        "none": {
            "advantages": ["Skips false positives on booleans"],
            "weaknesses": ["No detection at all"],
            "assumptions": ["Binary columns rarely have numeric 'outliers'"],
            "failure_modes": ["Encoded glitches disguised as strings/ints"],
            "guardrails": ["Consider data validation rules upstream"]
        },
    }
}

def _coerce_json(text: str) -> Dict[str, Any]:
    t = text.replace("```json", "").replace("```", "").strip()
    t = re.sub(r'^\s*//.*$', '', t, flags=re.MULTILINE)
    try:
        return json.loads(t)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", t)
        if match:
            cleaned = re.sub(r'^\s*//.*$', '', match.group(0), flags=re.MULTILINE)
            return json.loads(cleaned)
        raise

def generate_llm_json_report(profile: dict,
                             results: dict,
                             selection: dict,
                             provider: Literal["openai","ollama"] = "ollama",
                             model: Optional[str] = None,
                             temperature: float = 0.0) -> Dict[str, Any]:
    llm = load_llm(provider=provider, model=model, temperature=temperature)

    # System guidance: demand strict JSON and rich rationale
    system = (
        "You are a meticulous data quality expert. Return STRICT JSON only. "
        "Ground every decision in provided metrics (MAE/ACC/MAE_days, runtime), dtypes, and missing rates. "
        "Explicitly state: rationale, advantages, weaknesses/failure_modes, assumptions (incl. MNAR/leakage), "
        "alternatives considered & why rejected, and guardrails. "
        "If anomaly keys are provided, include a parallel analysis for detectors."
    )

    # Build the user payload with our KB to guide pros/cons
    user = {
        "task": "Generate comprehensive imputation + anomaly QA JSON report with explicit reasoning.",
        "profile": profile,                       # has n_rows, n_cols, columns[{name,dtype,missing_rate,cardinality?}]
        "results_per_method": results,            # per-column metrics for all tested methods
        "selection": selection,                   # {col: {method, metrics, dtype}}
        "method_kb": IMPUTATION_KB,
        "anomaly_kb": ANOMALY_KB,
        "anomaly_results": results.get("anomaly_results", {}),
        "anomaly_selection": results.get("anomaly_selection", {}),
        "artifacts": results.get("artifacts", {}),  # optional paths (your runner already adds some)
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
                "artifacts",
                "decision_log"
            ],
            # Require rich per-column structures:
            "per_column_shapes": {
                "anomaly_decisions": {
                    "keys": [
                        "column","dtype","chosen_detector","metrics",
                        "why_this_detector","advantages","weaknesses",
                        "assumptions","failure_modes","thresholds_or_params",
                        "alternatives_rejected","guardrails"
                    ]
                },
                "imputation_decisions": {
                    "keys": [
                        "column","dtype","missing_rate","chosen_imputer","metrics",
                        "why_this_imputer","advantages","weaknesses","assumptions",
                        "leakage_risk","mnar_note","data_cues_used",
                        "alternatives_rejected","guardrails"
                    ]
                }
            }
        }
    }

    prompt = f"{system}\n\nUser JSON:\n{json.dumps(user, ensure_ascii=False)}\n\n" \
             "Return a SINGLE JSON object with exactly these top-level keys:\n" \
             "- dataset_summary\n" \
             "- anomaly_decisions\n" \
             "- imputation_decisions\n" \
             "- comparative_analysis\n" \
             "- global_recommendations\n" \
             "- risks_and_assumptions\n" \
             "- next_steps\n" \
             "- artifacts\n" \
             "- decision_log\n\n" \
             "Where:\n" \
             "* anomaly_decisions: array of objects per column with the keys listed in constraints.per_column_shapes.anomaly_decisions.keys\n" \
             "* imputation_decisions: array of objects per column with the keys listed in constraints.per_column_shapes.imputation_decisions.keys\n" \
             "* comparative_analysis: show top-2 method/detector contrasts (metric deltas) when close.\n" \
             "* decision_log: brief, chronological bullet points of key decisions.\n" \
             "No prose outside JSON."

    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))
    data = _coerce_json(text)

    # Ensure all keys exist (robustness)
    for k in [
        "dataset_summary","anomaly_decisions","imputation_decisions",
        "comparative_analysis","global_recommendations","risks_and_assumptions",
        "next_steps","artifacts","decision_log"
    ]:
        if k not in data:
            data[k] = [] if k in ("anomaly_decisions","imputation_decisions","comparative_analysis","decision_log") else {}

    return data
