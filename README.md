# ReturnShield AI
<img width="1852" height="826" alt="image" src="https://github.com/user-attachments/assets/bb564fe3-221b-4d62-8f91-7e0dc6ea0f03" />

**Explainable, risk-based return-abuse detection for e-commerce merchants.**

Built for the **Razorpay Buildathon — Track 02: AI Risk Manager**.

[![Status](https://img.shields.io/badge/status-hackathon%20MVP-yellow)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Table of contents

- [Overview](#overview)
- [Why this track](#why-this-track)
- [Results](#results-held-out-test-set)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [API reference](#api-reference)
- [Dashboard](#dashboard)
- [Evaluation methodology](#evaluation-methodology)
- [Testing](#testing)
- [Demo](#demo)
- [Safety and fairness](#safety-and-fairness)
- [Tech stack](#tech-stack)
- [Known limitations / roadmap](#known-limitations--roadmap)
- [License](#license)

---

## Overview

ReturnShield AI predicts whether an e-commerce return request may involve **product
swapping** or a **false-defect claim**, and recommends one of three proportionate
actions: **approve**, **verify**, or **manual review**. It combines order history,
logistics evidence, parcel weight, serial/SKU verification, customer behavior, and
**linked-account network analysis** into a single explainable risk score.

It is explicitly a **decision-support system, not a judge**: a risk score is never
proof of fraud, and the system cannot automatically deny a refund or ban a customer.
Every score comes with a plain-English explanation and a human review path.

**Primary users:** D2C brands, e-commerce operations teams, warehouse teams, and
fraud analysts.

## Why this track

Track 02 (AI Risk Manager) asks builders to "stop the merchant losing money to
fraud, returns and chargebacks" by building "a working detector, verifier or
auto-responder for one class of loss, with measured precision and recall on a
held-out test set." The judging bar is "honest metrics including false-positive
cost," and anything offense-capable is disqualified.

ReturnShield AI is built directly against that bar:

- **One class of loss:** product-swap and false-defect return abuse, including
  coordinated abuse rings
- **Held-out evaluation:** a customer-separated, time-based test split, never seen
  during training or threshold tuning
- **Honest metrics:** precision, recall, F1, PR-AUC, an explicit false-positive
  rate, and precision/recall at a fixed review capacity (top 5%) — not just
  favorable aggregate numbers
- **Defense-only:** approve / verify / manual review only. No auto-denial, no
  account bans, no payment blocking — including the abuse-network view, which is
  investigative only and has no action buttons

## Results (held-out test set)

| Policy | Expected loss | Savings vs. approve-all |
|---|---|---|
| Approve-all (do nothing) | ₹160,200 | — |
| Rule-based baseline only | ₹69,720 | ₹90,480 |
| **ReturnShield (blended model)** | **₹4,760** | **₹155,440** |

Full precision, recall, F1, PR-AUC, top-5%-capacity metrics, and calibration by
product category and payment type are in
[`reports/eval_report.md`](returnshield-ai/reports/eval_report.md), and the same
breakdowns are available live from the dashboard and the `/performance` endpoint.

## How it works

```
Synthetic data generation
        │
Feature engineering
(return frequency, weight/serial mismatch, claim history, linked-account graph)
        │
   ┌────┴────┐
Rule baseline   ML model (XGBoost)
   └────┬────┘
     Blended score
        │
  SHAP explanations
        │
 FastAPI scoring service
        │
   ┌────────┼────────┐
Streamlit    SQLite    Abuse-network
dashboard    storage   graph (NetworkX)
        │
Financial cost evaluation
```

Every prediction blends a transparent rule-based baseline (repeated returns, weight
mismatch, serial mismatch, repeated defect claims) with a trained XGBoost model, so
the system's output is never a pure black box. On top of per-case scoring, a
NetworkX graph links customers who share a hashed address, device, or payment
method, surfacing coordinated abuse rings that a single-case rule or model would
miss entirely.

## Project structure

```
returnshield-ai/
  src/
    data_gen.py       # synthetic order/return dataset generator, customer-separated splits
    rules.py          # rule-based baseline scoring
    features.py        # feature engineering + linked-account graph (build_linked_account_graph, get_customer_ring)
    model.py             # trained model + rule/model blend, risk bands, score_case()
    explain.py             # SHAP-based plain-English explanations
    api.py                   # FastAPI service
    db.py                      # SQLite persistence layer
    cost.py                      # financial cost model
    evaluate.py                    # held-out test evaluation, writes reports/eval_report.md
  dashboard/
    app.py                          # Streamlit dashboard (5 views)
  data/                                # generated datasets (gitignored)
  reports/
    eval_report.md                      # latest held-out evaluation results
  tests/
    test_data_gen.py
    test_model.py
    test_rules.py
    test_features.py                      # linked-account graph tests
    test_smoke.py
  DEMO_SCRIPT.md                            # walkthrough for a genuine + an abusive return case
  requirements.txt
  README.md
```

## Getting started

### Prerequisites

- Python 3.13+
- pip

### Setup

```bash
git clone https://github.com/Khushi-7git/Return-AI.git
cd Return-AI/returnshield-ai

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Generate the dataset

```bash
.venv/bin/python -c "from src.data_gen import generate_dataset; generate_dataset()"
```

This creates a customer-separated, time-based train/validation/test split (earliest
70% of customers → train, next 15% → validation, latest 15% → held-out test).

### Run the API

```bash
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Run the dashboard (in a second terminal)

```bash
.venv/bin/streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8008
```

On Replit, both are wired up as the **Project** workflow (Run button) — API on
`:8000`, dashboard on `:8008`.

## API reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/score/{return_id}` | Score a single return: `risk_score`, `risk_band`, `recommended_action`, `estimated_loss_if_approved`, `top_reasons` |
| `POST` | `/score/batch` | Score a CSV of returns at once |
| `GET` | `/queue` | Ranked return review queue |
| `GET` | `/case/{return_id}` | Full case detail: order/return evidence, customer history, SHAP reasons |
| `POST` | `/feedback/{return_id}` | Record a reviewer's approve/verify/manual-review decision |
| `GET` | `/performance` | Precision, recall, F1, PR-AUC, confusion matrix, false-positive rate, precision/recall at top-5% capacity, breakdowns by risk band, category, and payment type |
| `GET` | `/financial` | Expected loss under approve-all, rule-only, and blended-model policies |
| `GET` | `/network/rings` | All detected linked-account rings (ring_size ≥ 2), sorted by size, with shared attributes |
| `GET` | `/network/{customer_id}` | A single customer's ring: linked customers, shared attributes, and each member's return/abuse history |

**Example response — `POST /score/{return_id}`:**

```json
{
  "risk_score": 0.82,
  "risk_band": "High",
  "recommended_action": "manual_review",
  "estimated_loss_if_approved": 1500,
  "top_reasons": [
    "Returned item weight is 340g lighter than expected",
    "Serial number does not match the shipped unit",
    "2 prior false-defect claims in the last 90 days"
  ],
  "recommended_verification": "Inspect serial number and weight before refund"
}
```

**Example response — `GET /network/{customer_id}`:**

```json
{
  "customer_id": "CUST00842",
  "ring_size": 4,
  "linked_customers": ["CUST00219", "CUST00551", "CUST01003"],
  "shared_attributes": ["hashed_address", "hashed_device"]
}
```

## Dashboard

1. **Return queue** — sortable, ranked by risk score, with color-coded risk bands
   (green/amber/red) and a top-line KPI row (total cases, % high risk, total
   estimated loss exposure)
2. **Case detail** — order timeline, evidence, customer history, SHAP explanation,
   and manual override controls (approve / verify / manual review), which write the
   reviewer's decision back to SQLite as feedback
3. **Abuse network** — table of detected linked-account rings, a NetworkX graph
   visualization of a selected ring (nodes = customers, edges labeled by shared
   address/device/payment, colored by abuse history), and each member's return
   record. Read-only and investigative — no action buttons, consistent with the
   defense-only requirement
4. **Model performance** — precision, recall, F1, PR-AUC, confusion matrix,
   false-positive rate, precision/recall at top-5% review capacity, and breakdowns
   by risk band, category, and payment type
5. **Financial impact** — expected loss compared across approve-all, rule-only, and
   blended-model policies

## Evaluation methodology

- **Dataset:** synthetic, generated to include legitimate returns, product-swap
  abuse, false-defect claims, and simulated coordinated abuse rings (shared
  hashed address/device across customer IDs)
- **Split:** customer-separated, time-based — earliest 70% of customers for
  training, next 15% for validation (threshold tuning), latest 15% held out
  untouched for final test evaluation
- **Metrics:** precision, recall, F1, PR-AUC, confusion matrix, explicit
  false-positive rate, precision/recall at a fixed review capacity (top 5%), and
  calibration broken out by product category and payment type
- **Financial evaluation:** expected loss = (false positives × FP cost) + (false
  negatives × FN cost) + (manual reviews × review cost), compared against an
  approve-all baseline and a rule-only baseline
- **Network analysis:** connected components in the linked-account graph are
  reported separately (ring size, shared attributes) but are not folded into the
  precision/recall metrics above — they're investigative signal, not a labeled
  classification task

## Testing

```bash
.venv/bin/pytest
```

Covers dataset generation (`test_data_gen.py`), the rule baseline
(`test_rules.py`), the trained model (`test_model.py`), the linked-account graph
(`test_features.py`), and an end-to-end smoke test (`test_smoke.py`).

## Demo

See [`DEMO_SCRIPT.md`](returnshield-ai/DEMO_SCRIPT.md) for a walkthrough using one
clearly legitimate return and one clearly abusive return, covering case detail,
the abuse-network graph, SHAP explanation, recommended action, and the
financial-impact panel.

## Safety and fairness

- Approve / verify / manual review only — no automatic refund denial, account ban,
  or payment blocking, including in the abuse-network view
- No caste, religion, gender, disability, or other sensitive attributes used
- Pincode, COD status, and address never independently determine risk — only in
  combination with other signals
- Every score ships with an explanation and a human-review/appeal path
- Missing evidence is treated as uncertainty, not proof of fraud
- Model version, threshold, input data, and decision timestamp are logged with
  every prediction

## Tech stack

**Backend / ML:** Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, NetworkX
**Serving:** FastAPI, uvicorn
**Dashboard:** Streamlit, matplotlib (network graph rendering)
**Storage:** SQLAlchemy / SQLite
**Testing:** pytest
**Hosting (dev):** Replit

## Known limitations / roadmap

- Trained on synthetic data — not validated against real merchant transaction data
- Image-similarity verification (shipped vs. returned product photos) is an
  optional PRD extension not required for the MVP scope and not yet built
- Risk-band thresholds (Low <0.40, Medium 0.40–0.74, High ≥0.75) are tuned on the
  validation set for this dataset and are configurable, not universally optimal
- The abuse-network graph is rebuilt from the current dataset and does not yet
  persist ring history across dataset regenerations
- Not yet integrated with a real payment or logistics provider — this is a
  decision-support prototype, not a production system

## License

MIT — hackathon submission for the Razorpay Buildathon, AI Risk Manager track.
