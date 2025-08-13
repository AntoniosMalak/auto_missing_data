
# Imputation Agent (Anomalies + Imputation) — JSON-First

A small, production-friendly toolkit that:
- Profiles your CSV dataset
- Benchmarks imputation methods per column and chooses the **best** one
- (Optionally) benchmarks per-column **anomaly detectors** and explains **why** each was chosen
- Generates a **strict JSON** LLM report that you can consume programmatically (no prose files required)
- Saves clean artifacts (imputed CSV, reports) with stable paths

> Built around `pandas`, `scikit-learn`, and LangChain providers (OpenAI or Ollama).

---

## Features

- **Strict JSON reporting**: `llm_report.json` contains:
  - `dataset_summary`
  - `anomaly_decisions` (per-column detector, metrics, rationale)
  - `imputation_decisions` (per-column imputer, metrics, rationale)
  - `comparative_analysis` (when top-2 methods are close)
  - `global_recommendations`, `risks_and_assumptions`, `next_steps`
  - `artifacts` (paths for output files)
- **Robust dtype handling**: boolean/categorical casting for `SimpleImputer` + dtype restoration.
- **Anomaly detectors included**: Z-Score, IQR, Isolation Forest, LOF, rare/unexpected categories, datetime Z-Score.
- **Agent mode** with LangGraph orchestrating profile → pipeline → JSON report.
- **No JSONDecodeError**: selections are handled as dicts; tools accept dicts or JSON strings safely.

---

## Project Layout

```
__init__.py
agent.py
anomaly_evaluate.py
anomaly_methods.py
anomaly_runner.py
cli.py
config.py
evaluate.py
llm.py
llm_report.py
methods.py
profiling.py
report.py
runner.py
selector.py
README.md
```

---

## Requirements

- Python **3.10+**
- Recommended: a virtual environment

**Python packages** (install via `pip`):
```
pandas
numpy
scikit-learn
joblib
typer
rich
pydantic
python-dotenv
langchain
langchain-openai
langchain-ollama
langgraph
```
You can create a `requirements.txt` with the above lines.

---

## Installation

```bash
# 1) (Optional) Create and activate a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt
# or install the list under Requirements
```

> If you plan to use **OpenAI**, set `OPENAI_API_KEY` in your environment.
>
> If you plan to use **Ollama**, make sure the Ollama daemon is running and the model exists (e.g., `ollama pull llama3.1:8b`).

---

## Quickstart

### Deterministic pipeline (no LLM)

```bash
python -m imputation_agent.cli run --csv path/to/data.csv --out outputs
```

Outputs:
- `outputs/imputed.csv`
- `outputs/all_methods_report.json`
- `outputs/imputation_report.json`
- `outputs/imputation_report.md`

### Deterministic pipeline + JSON LLM report

```bash
python -m imputation_agent.cli run --csv path/to/data.csv --out outputs   --llm-report --provider openai --model gpt-4o-mini
```
or with Ollama:
```bash
python -m imputation_agent.cli run --csv path/to/data.csv --out outputs   --llm-report --provider ollama --model "llama3.1:8b"
```
Adds:
- `outputs/llm_report.json`  ← a single, strict JSON object with full rationale.

### Agent Orchestration (profile → pipeline → LLM JSON)

```bash
python -m imputation_agent.cli plan-run   --csv path/to/data.csv   --out outputs   --provider openai   --model gpt-4o-mini   --temperature 0.0
```
Produces `outputs/llm_report.json` with the final agent output (pure JSON).

```bash
python -m imputation_agent.cli run-with-anomaly --csv path/to/data.csv --out outputs --seeds "1,2,3" --llm-report --provider openai --model gpt-4o-mini --temperature 0.0
```

```bash
python -m imputation_agent.cli run-with-anomaly --csv path/to/data.csv --out outputs --seeds "1,2,3" --llm-report --provider ollama --model "llama3.1:8b" --temperature 0.0
```

---

## JSON Outputs

### `imputation_report.json` (from `runner.py`)
```json
{
  "n_rows": 1234,
  "n_cols": 27,
  "selection": {
    "amount": {
      "method": "median",
      "metrics": { "MAE": 12.34, "runtime_sec": 0.01 },
      "dtype": "numeric"
    }
  },
  "timestamp": 1720000000.0,
  "artifacts": {
    "imputed_csv": "outputs/imputed.csv",
    "all_methods_report": "outputs/all_methods_report.json",
    "report_json": "outputs/imputation_report.json"
  }
}
```

### `all_methods_report.json`
All per-column candidate methods with averaged metrics:
```json
{
  "amount": {
    "mean": { "MAE": 15.21, "runtime_sec": 0.01 },
    "median": { "MAE": 12.34, "runtime_sec": 0.01 },
    "knn": { "MAE": 18.99, "runtime_sec": 2.36 }
  }
}
```

### `llm_report.json` (strict, single JSON object)
Keys:
- `dataset_summary`
- `anomaly_decisions` (if anomaly info present)
- `imputation_decisions`
- `comparative_analysis`
- `global_recommendations`
- `risks_and_assumptions`
- `next_steps`
- `artifacts`

Example (abridged):
```json
{
  "dataset_summary": { "rows": 1234, "columns": 27, "notes": [] },
  "anomaly_decisions": {
    "amount": { "chosen": "iqr", "metrics": { "precision": 0.8, "recall": 0.7, "f1": 0.75 }, "why": "Strong F1; robust to skew." }
  },
  "imputation_decisions": {
    "amount": { "chosen": "median", "metrics": { "MAE": 12.34 }, "why": "Less sensitive to outliers than mean." }
  },
  "comparative_analysis": {},
  "global_recommendations": [],
  "risks_and_assumptions": [],
  "next_steps": [],
  "artifacts": {
    "imputed_csv": "outputs/imputed.csv",
    "all_methods_report": "outputs/all_methods_report.json",
    "report_json": "outputs/imputation_report.json"
  }
}
```

---

## Environment Variables

- `OPENAI_API_KEY` — required if `--provider openai`.
- For Ollama, ensure the server is running locally and the model exists.

---

## Troubleshooting

- **`JSONDecodeError` during LLM report**  
  This build removes the root cause by passing `selection` as a dict end-to-end and by hardening `tool_llm_report` to accept dicts/JSON strings. If your model still returns fenced code blocks, the parser strips fences and extracts the JSON object.

- **Boolean column errors with `SimpleImputer`**  
  We cast boolean/categorical data to `object` for imputation, then restore boolean dtype.

- **Sparse or all-missing columns**  
  If a column has <= 1 non-missing value, it’s skipped for scoring; it will still appear in summaries.

---

## Extending

- Add new imputers in `methods.py` and update `config.py` to include them.
- Add new anomaly detectors in `anomaly_methods.py` and wire them into `DETECTORS`.

---

## License

MIT (or your preferred license).
