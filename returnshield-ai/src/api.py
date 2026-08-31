"""FastAPI service for scoring, queue data, analytics, and review feedback."""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from math import ceil
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .cost import financial_impact
from .db import (
    get_customer_history,
    get_latest_feedback,
    get_order_for_return,
    get_return,
    initialize_database,
    list_cases,
    save_feedback,
    save_score,
)
from .model import score_case, score_cases


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="ReturnShield AI", lifespan=lifespan)
TOP_CAPACITY_PCT = 5


class FeedbackRequest(BaseModel):
    decision: Literal["approve", "verify", "manual_review"]


def _score_return(return_id: str) -> dict[str, Any]:
    return_record = get_return(return_id)
    if return_record is None:
        raise HTTPException(status_code=404, detail=f"Unknown return_id: {return_id}")
    order = get_order_for_return(return_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"No order found for return_id: {return_id}")
    try:
        payload = score_case(str(order["order_id"]))
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    save_score(return_id, str(order["order_id"]), payload)
    return payload


def _ranked_queue() -> list[dict[str, Any]]:
    scores = {row["return_id"]: row for row in score_cases()}
    queue: list[dict[str, Any]] = []
    for case in list_cases():
        scored = scores.get(case["return_id"])
        if scored is None:
            continue
        queue.append(
            {
                "return_id": case["return_id"],
                "product": case["product_id"],
                "customer": case["customer_id"],
                "risk_score": scored["risk_score"],
                "risk_band": scored["risk_band"],
                "reason": case["reason"],
                "refund_amount": case["refund_amount"],
                "recommended_action": scored["recommended_action"],
            }
        )
        save_score(
            case["return_id"],
            case["order_id"],
            {
                **scored,
                "top_reasons": [],
                "recommended_verification": "Open the case for SHAP reasons and evidence.",
            },
        )
    return sorted(queue, key=lambda row: row["risk_score"], reverse=True)


@app.get("/")
def hello_world() -> dict[str, str]:
    """Keep the original smoke-test endpoint available."""
    return {"message": "Hello from ReturnShield AI"}


@app.get("/health")
def health() -> dict[str, str]:
    initialize_database()
    return {"status": "ok"}


@app.post("/score/batch")
async def score_batch(file: UploadFile = File(...)) -> dict[str, Any]:
    """Score return IDs or order IDs from an uploaded CSV and rank them."""
    try:
        frame = pd.read_csv(BytesIO(await file.read()))
    except Exception as error:
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid CSV") from error

    if "return_id" in frame.columns:
        identifiers = frame["return_id"].dropna().astype(str).tolist()
        identifier_kind = "return_id"
    elif "order_id" in frame.columns:
        order_ids = frame["order_id"].dropna().astype(str).tolist()
        identifiers = []
        for order_id in order_ids:
            matching = [
                case for case in list_cases() if str(case["order_id"]) == order_id
            ]
            if matching:
                identifiers.append(str(matching[0]["return_id"]))
        identifier_kind = "order_id"
    else:
        raise HTTPException(status_code=400, detail="CSV must contain return_id or order_id")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        try:
            payload = _score_return(identifier)
            return_record = get_return(identifier)
            results.append(
                {
                    "return_id": identifier,
                    "order_id": str(return_record["order_id"]) if return_record else "",
                    **payload,
                }
            )
        except HTTPException as error:
            errors.append({"identifier": identifier, "error": str(error.detail)})

    results.sort(key=lambda row: row["risk_score"], reverse=True)
    return {
        "results": results,
        "count": len(results),
        "errors": errors,
        "identifier_kind": identifier_kind,
    }


@app.post("/score/{return_id}")
def score_return(return_id: str) -> dict[str, Any]:
    """Score one return and persist its score in SQLite."""
    return _score_return(return_id)


@app.get("/queue")
def queue() -> list[dict[str, Any]]:
    """Return the ranked return queue for the dashboard."""
    return _ranked_queue()


@app.get("/case/{return_id}")
def case_detail(return_id: str) -> dict[str, Any]:
    """Return order, return, history, evidence, and detailed score data."""
    return_record = get_return(return_id)
    order = get_order_for_return(return_id)
    if return_record is None or order is None:
        raise HTTPException(status_code=404, detail=f"Unknown return_id: {return_id}")

    score = _score_return(return_id)
    history = get_customer_history(str(order["customer_id"]), return_id)
    expected_weight = float(order["expected_weight_g"])
    received_weight = float(return_record["received_weight_g"])
    evidence = {
        "expected_weight_g": expected_weight,
        "received_weight_g": received_weight,
        "weight_difference_g": round(expected_weight - received_weight, 1),
        "weight_match": abs(expected_weight - received_weight) / expected_weight <= 0.20,
        "serial_match": str(order["shipped_serial"]) == str(return_record["received_serial"]),
        "prior_return_count": len(history),
        "prior_reasons": [record["reason"] for record in history],
    }
    timeline = [
        {
            "event": "order_placed",
            "timestamp": order["timestamp"],
            "description": f"{order['product_id']} ordered",
        },
        {
            "event": "return_requested",
            "timestamp": return_record["timestamp"],
            "description": str(return_record["reason"]).replace("_", " "),
        },
    ]
    return {
        "return_id": return_id,
        "order": order,
        "return": return_record,
        "timeline": timeline,
        "evidence": evidence,
        "history": history,
        "score": score,
        "latest_feedback": get_latest_feedback(return_id),
    }


@app.post("/feedback/{return_id}")
def feedback(return_id: str, request: FeedbackRequest) -> dict[str, Any]:
    """Persist a reviewer's manual decision in SQLite."""
    if get_return(return_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown return_id: {return_id}")
    return save_feedback(return_id, request.decision)


def _metrics_by_group(
    cases: list[dict[str, Any]],
    group_key: str,
    fixed_groups: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Calculate classification metrics for each sorted group of cases."""
    grouped_cases: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        group_value = case.get(group_key)
        group_name = "Unknown" if group_value is None else str(group_value)
        grouped_cases.setdefault(group_name, []).append(case)

    group_names = fixed_groups or sorted(grouped_cases)
    metrics: dict[str, dict[str, Any]] = {}
    for group_name in group_names:
        group_cases = grouped_cases.get(group_name, [])
        if not group_cases:
            metrics[group_name] = {
                "count": 0,
                "abuse_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "pr_auc": None,
            }
            continue

        group_true = [case["confirmed_abuse_label"] for case in group_cases]
        group_pred = [int(case["risk_score"] >= 0.5) for case in group_cases]
        metrics[group_name] = {
            "count": len(group_cases),
            "abuse_rate": round(sum(group_true) / len(group_true), 6),
            "precision": precision_score(group_true, group_pred, zero_division=0),
            "recall": recall_score(group_true, group_pred, zero_division=0),
            "f1": f1_score(group_true, group_pred, zero_division=0),
            "pr_auc": (
                average_precision_score(
                    group_true,
                    [case["risk_score"] for case in group_cases],
                )
                if len(set(group_true)) > 1
                else None
            ),
        }
    return metrics


@app.get("/performance")
def performance(capacity_pct: int = TOP_CAPACITY_PCT) -> dict[str, Any]:
    """Return classification, calibration, and risk-band metrics."""
    cases = score_cases()
    y_true = [case["confirmed_abuse_label"] for case in cases]
    y_score = [case["risk_score"] for case in cases]
    top_capacity_count = max(1, ceil(len(cases) * capacity_pct / 100))
    ranked_cases = sorted(cases, key=lambda case: case["risk_score"], reverse=True)
    top_capacity_cases = ranked_cases[:top_capacity_count]
    top_capacity_true = [case["confirmed_abuse_label"] for case in ranked_cases]
    top_capacity_pred = [
        int(index < top_capacity_count) for index in range(len(ranked_cases))
    ]
    top_capacity_precision = precision_score(
        top_capacity_true,
        top_capacity_pred,
        zero_division=0,
    )
    top_capacity_recall = recall_score(
        top_capacity_true,
        top_capacity_pred,
        zero_division=0,
    )
    y_pred = [int(score >= 0.5) for score in y_score]
    precision, recall, f1 = (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )
    calibration_actual, calibration_predicted = calibration_curve(
        y_true, y_score, n_bins=10, strategy="quantile"
    )
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp = confusion[0][0], confusion[0][1]
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    band_metrics = _metrics_by_group(cases, "risk_band", ["Low", "Medium", "High"])
    category_metrics = _metrics_by_group(cases, "category")
    payment_type_metrics = _metrics_by_group(cases, "payment_type")

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": average_precision_score(y_true, y_score),
            "false_positive_rate": false_positive_rate,
        },
        "top_capacity_metrics": {
            "capacity_pct": capacity_pct,
            "cases_reviewed": len(top_capacity_cases),
            "precision": top_capacity_precision,
            "recall": top_capacity_recall,
        },
        "confusion_matrix": confusion,
        "calibration": {
            "predicted_probability": calibration_predicted.tolist(),
            "observed_rate": calibration_actual.tolist(),
        },
        "by_risk_band": band_metrics,
        "by_category": category_metrics,
        "by_payment_type": payment_type_metrics,
    }


@app.get("/financial")
def financial() -> dict[str, Any]:
    """Return policy loss comparisons for the financial impact view."""
    return financial_impact(score_cases())