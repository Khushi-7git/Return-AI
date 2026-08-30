# returnshield-ai

Starter Python project for ReturnShield AI.

This repository currently contains only the project skeleton plus minimal
environment checks:

- A `GET /` FastAPI endpoint returning a hello-world response.
- A Streamlit page that displays a hello-world message.
- Empty module placeholders for future data, feature, rules, model,
  explanation, and cost work.

## Setup

```bash
cd returnshield-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the API

```bash
uvicorn src.api:app --reload
```

The endpoint is available at <http://127.0.0.1:8000/>.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Verify the environment

```bash
python -c "import pandas, numpy, sklearn, xgboost, shap, networkx, streamlit, fastapi, uvicorn, sqlalchemy, faker, pytest; print('All imports clean')"
pytest
```