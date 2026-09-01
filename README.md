# ReturnShield AI

**Razorpay Buildathon — Track 02: AI Risk Manager**

A defense-only return-risk scoring system for e-commerce merchants. Predicts whether
a return may involve product swapping or a false-defect claim, and recommends
**approve / verify / manual review** — never an automatic refund denial or account ban.

## Why this track

Track 02 asks for "a working detector, verifier or auto-responder for one class of
loss, with measured precision and recall on a held-out test set," judged on "honest
metrics including false-positive cost," and disqualifies anything offense-capable.
ReturnShield AI is a return-risk scorer built specifically against that bar: every
prediction ships with an explanation, every action is a proportionate verification
step (not a punishment), and evaluation runs on a customer-separated, time-based
held-out test set — down to precision/recall at a fixed review capacity and an
explicit false-positive rate, not just aggregate numbers.

## Results (held-out test set)

| Policy | Expected loss | Savings vs. approve-all |
|---|---|---|
| Approve-all (baseline) | ₹160,200 | — |
| Rule-based only | ₹69,720 | ₹90,480 |
| **ReturnShield (blended model)** | **₹4,760** | **₹155,440** |

Full precision, recall, F1, PR-AUC, calibration-by-category, calibration-by-payment-
type, and top-capacity figures are in
[`reports/eval_report.md`](returnshield-ai/reports/eval_report.md) — and the same
breakdowns are live in the dashboard's Model Performance view, not just the offline
report.

## How it works

```
Synthetic data generation
        │
Feature engineering (return frequency, weight/serial mismatch, claim history, linked accounts)
        │
   ┌────┴────┐
Rule baseline   ML model (blended)
   └────┬────┘
        │
  SHAP explanations
        │
 FastAPI scoring service
        │
   ┌────┴────┐
Streamlit dashboard   SQLite storage
        │
Financial cost evaluation
```

Every score is a blend of a transparent rule-based baseline (repeated returns, weight
mismatch, serial mismatch, repeated defect claims) and a trained ML model, so the
system never depends solely on a black box.

## Project structure

```
returnshield-ai/
  src/
    data_gen.py     # synthetic order/return dataset generator, customer-separated splits
    rules.py        # rule-based baseline scoring
    features.py     # feature engineering (behavioral, logistics, linked-account)
    model.py         # trained model + rule/model blend, risk bands, score_case()
    explain.py       # SHAP-based plain-English explanations
    api.py            # FastAPI service
    db.py              # SQLite persistence
    cost.py            # financial cost model
    evaluate.py        # held-out test evaluation, writes reports/eval_report.md
  dashboard/
    app.py             # Streamlit dashboard
  data/                 # generated datasets (gitignored)
  reports/
    eval_report.md       # latest held-out evaluation results
  tests/                  # pytest suite
  DEMO_SCRIPT.md            # walkthrough for a genuine + an abusive return case
  requirements.txt
```

## Running it

```bash
cd returnshield-ai
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# generate the synthetic dataset
.venv/bin/python -c "from src.data_gen import generate_dataset; generate_dataset()"

# start the API
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000

# in a second terminal, start the dashboard
.venv/bin/streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8008
```

On Replit, both are wired up as the **Project** workflow (Run button) — API on
`:8000`, dashboard on `:8008`.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /score/{return_id}` | Score a single return: risk_score, risk_band, recommended_action, top_reasons |
| `POST /score/batch` | Score a CSV of returns at once |
| `GET /queue` | Ranked return review queue |
| `GET /case/{return_id}` | Full case detail: evidence, history, SHAP reasons |
| `POST /feedback/{return_id}` | Record a reviewer's approve/verify/manual-review decision |
| `GET /performance` | Precision, recall, F1, PR-AUC, confusion matrix, false-positive rate, precision/recall at top-5% review capacity, breakdowns by risk band, product category, and payment type |
| `GET /financial` | Expected loss vs. approve-all and rule-only baselines |

## Dashboard views

1. **Return queue** — sortable, ranked by risk score, with color-coded risk bands
   (green/amber/red) and a top-line KPI row (total cases, % high risk, total
   estimated loss exposure)
2. **Case detail** — evidence, history, SHAP explanation, manual override
3. **Model performance** — precision, recall, F1, PR-AUC, confusion matrix,
   false-positive rate, precision/recall at top-5% review capacity, and breakdowns
   by risk band, category, and payment type
4. **Financial impact** — expected loss across policies (approve-all, rule-only,
   blended model)

## Evaluation metrics (per PRD requirements)

- Precision, recall, F1, PR-AUC — overall and by risk band
- **False-positive rate**, surfaced explicitly, not just derived from the confusion matrix
- **Precision/recall at top-5% review capacity** — answers "if the team can only
  review 5% of returns, how many abusers are caught"
- Calibration and performance by product category and payment type
- Expected financial loss vs. approve-all and rule-based baselines

## Safety and fairness

- Approve / verify / manual review only — no automatic refund denial or account ban
- No caste, religion, gender, or disability signals used
- Pincode, COD status, and address never independently determine risk
- Every score ships with explanations and a human-review path
- Missing evidence is treated as uncertainty, not proof of fraud
- Model version, threshold, and input data are logged with every decision

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, NetworkX, FastAPI, Streamlit,
SQLAlchemy/SQLite, pytest.
