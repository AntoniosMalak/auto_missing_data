
# Imputation Agent (Evaluation + Taxonomy + LLM JSON Reports)

Features:
- Load CSV, profile schema/missingness
- Try **all configured imputation techniques** per type
- Evaluate via mask-and-score (MAE for numeric, ACC for categorical/boolean, MAE_days for datetime)
- Choose best per column and impute full dataset
- Save **full comparison** (`all_methods_report.json`), **selection** (`imputation_report.json`), and **HTML-like Markdown** report
- Optional **LLM JSON report** using LangChain with **OpenAI or Ollama**

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Deterministic run (compare all techniques)
```bash
python -m imputation_agent.cli run --csv path/to/data.csv --out outputs --llm-report --provider ollama --model "llama3.1:8b"
```

## Agent run (OpenAI or Ollama)
```bash
# Ollama
python -m imputation_agent.cli plan-run --csv path/to/data.csv --out outputs --provider ollama --model "llama3.1:8b"

# OpenAI
export OPENAI_API_KEY=sk-...
python -m imputation_agent.cli plan-run --csv path/to/data.csv --out outputs --provider openai --model gpt-4o-mini
```

Outputs in `outputs/`:
- `imputed.csv`
- `all_methods_report.json`
- `imputation_report.json`
- `imputation_report.md`
- (optional) `llm_report.json`


## Anomaly Detection

Run per-column anomaly model selection and treatment:

```bash
python -m imputation_agent.cli anomaly --input-csv data.csv --out-dir out_anom
```
This will try multiple detectors per dtype, pick the best by F1 on synthetic anomalies, mark detected anomalies as NaN, then you can run the existing imputation pipeline.
