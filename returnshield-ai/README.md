# returnshield-ai

Starter Python project for ReturnShield AI.

The project currently includes a deterministic synthetic data generator and an
explainable rule-based baseline, plus minimal environment checks:

- A `GET /` FastAPI endpoint returning a hello-world response.
- A Streamlit page that displays a hello-world message.
- `src.data_gen.generate_dataset()` for orders, returns, abuse scenarios, and
  customer-separated train/validation/test assignments.
- `src.rules.score_rules()` for repeated returns, weight mismatch,
  serial/SKU mismatch, and repeated abuse-claim patterns.
- `src.explain.explain_prediction()` for plain-English SHAP reasons.
- `src.model.score_case()` for the PRD risk score, action, estimated loss, and
  recommended verification.
- Empty module placeholders for future feature and cost work.

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

## Generate the synthetic data

```bash
python -c "from src.data_gen import generate_dataset; generate_dataset()"
```

This writes `data/orders.csv`, `data/returns.csv`, and `data/split.csv`.
`generate_dataset()` returns the orders and returns DataFrames; the confirmed
abuse label is retained only in the returns data for evaluation.

## Score a return case

```bash
python -c "from src.model import score_case; print(score_case('ORD-0000002'))"
```

`score_case()` trains its small Random Forest lazily from the generated data
and combines its abuse probability with the rule score using a weighted blend:

```text
risk_score = rule_weight * rule_score + (1 - rule_weight) * ml_score
```

The default is a 50/50 blend. Set `RETURNSHIELD_RULE_WEIGHT` to a value from
`0` to `1` before calling `score_case()` to configure it. The order must have a
return record; orders without returns raise a clear error.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Verify the environment

```bash
python -c "import pandas, numpy, sklearn, xgboost, shap, networkx, streamlit, fastapi, uvicorn, sqlalchemy, faker, pytest; print('All imports clean')"
pytest
```