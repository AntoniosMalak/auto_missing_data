
# Imputation Agent (Python / LangChain)

End-to-end pipeline to:
1) Profile a CSV.
2) Try multiple imputation techniques.
3) Evaluate with mask-and-score.
4) Pick the best per column.
5) Impute full data.
6) Generate artifacts: `imputed.csv`, `imputation_report.json`, `imputation_report.md`.

## Modes
- **Deterministic (no LLM)** — run the pipeline directly.
- **Agent mode (LLM)** — choose **OpenAI** or **Ollama** local model to orchestrate tools.

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
=======
# 1) Create venv (recommended)
python -m venv venv && . venv/bin/activate  # on Windows: venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) Run the pipeline (no LLM needed)
python -m src.imputation_agent.cli run --csv path/to/your.csv --out outputs

# 4) (Optional) Use the planning Agent (requires OPENAI_API_KEY or another llm in .env)
## Agent with Ollama (local)
# Ensure Ollama is running and model is available:
# ollama pull llama3.1:8b-instruct
python -m imputation_agent.cli plan-run --csv path/to/your.csv --out outputs --provider ollama --model "llama3.1:8b-instruct"

## Agent with OpenAI (cloud)
export OPENAI_API_KEY=sk-...
python -m imputation_agent.cli plan-run --csv path/to/your.csv --out outputs --provider openai --model gpt-4o-mini
Outputs in `outputs/`: `imputed.csv`, `imputation_report.json`, `imputation_report.md`, `imputers.joblib`.

```
