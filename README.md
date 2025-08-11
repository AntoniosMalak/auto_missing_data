
# Imputation Agent (Python / LangChain optional)

End-to-end pipeline to:
1) Profile a CSV.
2) Try multiple imputation techniques.
3) Evaluate with mask-and-score.
4) Pick the best per column.
5) Impute full data.
6) Generate artifacts: `imputed.csv`, `imputation_report.json`, `imputation_report.md`.

You can run it **without any LLM/agent** via CLI, or enable a light **LangChain agent** that orchestrates tools.

## Quick start

```bash
# 1) Create venv (recommended)
python -m venv venv && . venv/bin/activate  # on Windows: venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) Run the pipeline (no LLM needed)
python -m imputation_agent.cli run --csv path/to/your.csv --out outputs

# 4) (Optional) Use the planning Agent (requires OPENAI_API_KEY or another llm in .env)
python -m imputation_agent.cli plan-run --csv path/to/your.csv --out outputs
```

## Outputs
- `imputed.csv` — fully imputed dataset
- `imputation_report.json` — machine-readable summary
- `imputation_report.md` — human-readable summary
- `imputers.joblib` — persisted fitted imputers

## Config
See `src/imputation_agent/config.py` for defaults. Override via CLI flags.

## Notes
- For very large datasets, start with the baseline methods (median/mode) and limit KNN/MICE via `--max-candidates` and `--budget-seeds`.
- Agent mode is optional. The deterministic pipeline works standalone.
