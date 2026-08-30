"""Model training and case scoring for ReturnShield AI."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .explain import explain_prediction
from .rules import score_rules


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEATURE_COLUMNS = [
    "price",
    "expected_weight_g",
    "received_weight_g",
    "weight_difference_g",
    "weight_mismatch_ratio",
    "days_after_delivery",
    "refund_ratio",
    "serial_mismatch",
    "sku_mismatch",
    "packaging_tampered",
    "claim_is_defect",
    "claim_is_swap",
    "repeat_returns_90d",
    "customer_return_count",
    "prior_false_defect_claims",
    "prior_swap_claims",
    "address_customer_count",
    "device_customer_count",
]


def _value(record: Any, key: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(key, default)
    if isinstance(record, pd.Series):
        return record.get(key, default)
    return getattr(record, key, default)


def _records(history: Any) -> list[Any]:
    if history is None:
        return []
    if isinstance(history, pd.DataFrame):
        return history.to_dict("records")
    if isinstance(history, Mapping):
        nested = history.get("returns")
        return _records(nested) if nested is not None else [history]
    if isinstance(history, Iterable) and not isinstance(history, (str, bytes)):
        return list(history)
    return [history]


def _timestamp(record: Any) -> pd.Timestamp | None:
    value = _value(record, "timestamp")
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _claim_family(reason: Any) -> str | None:
    normalized = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "defective",
        "defect",
        "damaged",
        "not_working",
        "false_defect",
        "item_not_working",
    }:
        return "false_defect"
    if normalized in {
        "wrong_item_received",
        "wrong_item",
        "item_mismatch",
        "product_swap",
        "swap",
    }:
        return "swap"
    return None


def _prior_history(
    current_return: Any,
    history: Any,
) -> list[Any]:
    current_timestamp = _timestamp(current_return)
    current_return_id = _value(current_return, "return_id")
    prior: list[Any] = []
    for record in _records(history):
        if current_return_id is not None and _value(record, "return_id") == current_return_id:
            continue
        record_timestamp = _timestamp(record)
        if (
            current_timestamp is not None
            and record_timestamp is not None
            and record_timestamp > current_timestamp
        ):
            continue
        prior.append(record)
    return prior


def _relative_difference(expected: Any, received: Any) -> float:
    try:
        expected_float = float(expected)
        received_float = float(received)
    except (TypeError, ValueError):
        return 0.0
    if expected_float <= 0:
        return 0.0
    return (expected_float - received_float) / expected_float


def _case_features(
    order: Any,
    current_return: Any,
    history: Any,
    address_customer_counts: Mapping[str, int],
    device_customer_counts: Mapping[str, int],
) -> dict[str, float]:
    prior = _prior_history(current_return, history)
    current_timestamp = _timestamp(current_return)

    recent_prior = []
    for record in prior:
        record_timestamp = _timestamp(record)
        if (
            current_timestamp is None
            or record_timestamp is None
            or (current_timestamp - record_timestamp).total_seconds() <= 90 * 86400
        ):
            recent_prior.append(record)

    prior_families = [_claim_family(_value(record, "reason")) for record in prior]
    expected_weight = _value(order, "expected_weight_g", 0.0)
    received_weight = _value(current_return, "received_weight_g", 0.0)
    price = float(_value(order, "price", 0.0) or 0.0)
    refund_amount = float(_value(current_return, "refund_amount", 0.0) or 0.0)

    shipped_serial = _value(order, "shipped_serial")
    received_serial = _value(current_return, "received_serial")
    serial_mismatch = int(
        shipped_serial not in (None, "")
        and received_serial not in (None, "")
        and str(shipped_serial) != str(received_serial)
    )

    order_sku = _value(order, "product_id", _value(order, "sku"))
    return_sku = _value(current_return, "product_id", _value(current_return, "sku"))
    sku_mismatch = int(
        order_sku not in (None, "")
        and return_sku not in (None, "")
        and str(order_sku) != str(return_sku)
    )

    try:
        days_after_delivery = float(_value(current_return, "days_after_delivery", 0.0) or 0.0)
    except (TypeError, ValueError):
        days_after_delivery = 0.0

    customer_id = str(_value(order, "customer_id", ""))
    address = str(_value(order, "hashed_address", ""))
    device = str(_value(order, "hashed_device", ""))
    packaging = str(_value(current_return, "packaging_condition", "")).lower()
    current_family = _claim_family(_value(current_return, "reason"))

    return {
        "price": price,
        "expected_weight_g": float(expected_weight or 0.0),
        "received_weight_g": float(received_weight or 0.0),
        "weight_difference_g": float(expected_weight or 0.0) - float(received_weight or 0.0),
        "weight_mismatch_ratio": _relative_difference(expected_weight, received_weight),
        "days_after_delivery": days_after_delivery,
        "refund_ratio": refund_amount / price if price > 0 else 0.0,
        "serial_mismatch": float(serial_mismatch),
        "sku_mismatch": float(sku_mismatch),
        "packaging_tampered": float(packaging in {"tampered", "damaged", "missing"}),
        "claim_is_defect": float(current_family == "false_defect"),
        "claim_is_swap": float(current_family == "swap"),
        "repeat_returns_90d": float(len(recent_prior)),
        "customer_return_count": float(len(prior)),
        "prior_false_defect_claims": float(prior_families.count("false_defect")),
        "prior_swap_claims": float(prior_families.count("swap")),
        "address_customer_count": float(address_customer_counts.get(address, 1)),
        "device_customer_count": float(device_customer_counts.get(device, 1)),
    }


def _build_feature_frame(
    orders: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build training features without using the abuse ground-truth label."""
    if orders.empty or returns.empty:
        raise ValueError("orders.csv and returns.csv must both contain data")

    order_lookup = orders.set_index("order_id", drop=False)
    address_customer_counts = (
        orders.groupby("hashed_address")["customer_id"].nunique().astype(int).to_dict()
    )
    device_customer_counts = (
        orders.groupby("hashed_device")["customer_id"].nunique().astype(int).to_dict()
    )

    joined = returns.merge(
        orders[["order_id", "customer_id"]],
        on="order_id",
        how="inner",
    ).sort_values(["customer_id", "timestamp", "return_id"], kind="stable")
    customer_history: dict[str, list[dict[str, Any]]] = {}
    feature_rows: dict[str, dict[str, float]] = {}

    for customer_id, customer_returns in joined.groupby("customer_id", sort=False):
        prior: list[dict[str, Any]] = []
        for return_record in customer_returns.to_dict("records"):
            order = order_lookup.loc[return_record["order_id"]]
            feature_rows[return_record["return_id"]] = _case_features(
                order,
                return_record,
                prior,
                address_customer_counts,
                device_customer_counts,
            )
            prior.append(return_record)
        customer_history[str(customer_id)] = prior

    missing_return_ids = set(returns["return_id"]) - set(feature_rows)
    if missing_return_ids:
        raise ValueError(f"Returns reference missing orders: {sorted(missing_return_ids)[:3]}")

    return pd.DataFrame.from_dict(feature_rows, orient="index")[FEATURE_COLUMNS].fillna(0.0)


@dataclass(frozen=True)
class _ScoringContext:
    orders: pd.DataFrame
    returns: pd.DataFrame
    returns_with_customers: pd.DataFrame
    features: pd.DataFrame
    model: RandomForestClassifier
    address_customer_counts: dict[str, int]
    device_customer_counts: dict[str, int]


@lru_cache(maxsize=1)
def _get_scoring_context() -> _ScoringContext:
    """Load data and lazily train the baseline ML model once per process."""
    orders_path = DATA_DIR / "orders.csv"
    returns_path = DATA_DIR / "returns.csv"
    if not orders_path.exists() or not returns_path.exists():
        raise FileNotFoundError(
            "Generated data is missing. Run "
            "`from src.data_gen import generate_dataset; generate_dataset()` first."
        )

    orders = pd.read_csv(orders_path)
    returns = pd.read_csv(returns_path)
    if "confirmed_abuse_label" not in returns:
        raise ValueError("returns.csv must contain confirmed_abuse_label for model training")

    features = _build_feature_frame(orders, returns)
    labels = returns.set_index("return_id").loc[features.index, "confirmed_abuse_label"].astype(int)
    if labels.nunique() < 2:
        raise ValueError("Training data must contain both abuse-label classes")

    model = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features, labels)

    returns_with_customers = returns.merge(
        orders[["order_id", "customer_id"]],
        on="order_id",
        how="inner",
    )
    address_customer_counts = (
        orders.groupby("hashed_address")["customer_id"].nunique().astype(int).to_dict()
    )
    device_customer_counts = (
        orders.groupby("hashed_device")["customer_id"].nunique().astype(int).to_dict()
    )
    return _ScoringContext(
        orders=orders,
        returns=returns,
        returns_with_customers=returns_with_customers,
        features=features,
        model=model,
        address_customer_counts=address_customer_counts,
        device_customer_counts=device_customer_counts,
    )


def _positive_class_probability(model: RandomForestClassifier, features: pd.DataFrame) -> float:
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
    return float(probabilities[positive_index])


def _rule_weight() -> float:
    raw_value = os.getenv("RETURNSHIELD_RULE_WEIGHT", "0.50")
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError("RETURNSHIELD_RULE_WEIGHT must be a number from 0 to 1") from error
    if not 0 <= value <= 1:
        raise ValueError("RETURNSHIELD_RULE_WEIGHT must be a number from 0 to 1")
    return value


def _recommended_verification(triggered_rules: list[str]) -> str:
    checks: list[str] = []
    if "serial_mismatch" in triggered_rules or "sku_mismatch" in triggered_rules:
        checks.append("verify the returned serial/SKU against the shipment")
    if "weight_mismatch" in triggered_rules:
        checks.append("weigh the item against the expected shipment weight")
    if (
        "repeated_returns_90d" in triggered_rules
        or "repeated_false_defect_claims" in triggered_rules
        or "repeated_swap_claims" in triggered_rules
    ):
        checks.append("review recent customer and device/address return history")
    if not checks:
        checks.append("verify the returned item condition and order details")
    return " and ".join(checks).capitalize() + "."


def risk_band_and_action(risk_score: float) -> tuple[str, str]:
    """Map a 0-1 risk score to the PRD band and recommended action."""
    if risk_score < 0.33:
        return "Low", "approve"
    if risk_score < 0.66:
        return "Medium", "verify"
    return "High", "manual_review"


def _case_components(
    context: _ScoringContext,
    order_id: str,
    ml_score_override: float | None = None,
) -> tuple[
    pd.Series,
    pd.Series,
    list[dict[str, Any]],
    float,
    list[str],
    pd.DataFrame,
    float,
    float,
    str,
    str,
]:
    """Calculate shared rule/ML components without running SHAP."""
    matching_orders = context.orders.loc[context.orders["order_id"].astype(str) == str(order_id)]
    if matching_orders.empty:
        raise KeyError(f"Unknown order_id: {order_id}")
    order = matching_orders.iloc[0]

    matching_returns = context.returns.loc[
        context.returns["order_id"].astype(str) == str(order_id)
    ].copy()
    if matching_returns.empty:
        raise KeyError(f"No return found for order_id: {order_id}")
    matching_returns["timestamp"] = pd.to_datetime(matching_returns["timestamp"], errors="coerce")
    current_return = matching_returns.sort_values("timestamp", kind="stable").iloc[-1]

    customer_returns = context.returns_with_customers.loc[
        context.returns_with_customers["customer_id"] == order["customer_id"]
    ].copy()
    history = customer_returns.loc[
        customer_returns["return_id"] != current_return["return_id"]
    ].to_dict("records")
    rule_score, triggered_rules = score_rules(order, current_return, history)

    feature_row = _case_features(
        order,
        current_return,
        history,
        context.address_customer_counts,
        context.device_customer_counts,
    )
    feature_frame = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    ml_score = (
        ml_score_override
        if ml_score_override is not None
        else _positive_class_probability(context.model, feature_frame)
    )
    rule_weight = _rule_weight()
    risk_score = min(1.0, max(0.0, rule_weight * rule_score + (1 - rule_weight) * ml_score))
    risk_band, recommended_action = risk_band_and_action(risk_score)
    return (
        order,
        current_return,
        history,
        rule_score,
        triggered_rules,
        feature_frame,
        ml_score,
        risk_score,
        risk_band,
        recommended_action,
    )


@lru_cache(maxsize=4)
def _score_cases_cached(rule_weight: float) -> tuple[dict[str, Any], ...]:
    """Return lightweight scores for every stored return.

    This is used by queue and analytics endpoints and intentionally skips SHAP
    text generation so a whole dataset can be ranked efficiently. Individual
    ``score_case`` calls include the full top-reason explanation.
    """
    context = _get_scoring_context()
    all_features = context.features.loc[:, FEATURE_COLUMNS]
    all_probabilities = context.model.predict_proba(all_features)
    classes = list(context.model.classes_)
    positive_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
    ml_scores = dict(
        zip(
            context.features.index,
            all_probabilities[:, positive_index].astype(float),
        )
    )
    results: list[dict[str, Any]] = []
    for return_record in context.returns.to_dict("records"):
        components = _case_components(
            context,
            str(return_record["order_id"]),
            ml_score_override=ml_scores[str(return_record["return_id"])],
        )
        (
            _order,
            current_return,
            _history,
            rule_score,
            triggered_rules,
            _feature_frame,
            ml_score,
            risk_score,
            risk_band,
            recommended_action,
        ) = components
        refund_amount = float(current_return.get("refund_amount", 0.0) or 0.0)
        results.append(
            {
                "return_id": str(current_return["return_id"]),
                "order_id": str(current_return["order_id"]),
                "rule_score": float(round(rule_score, 6)),
                "ml_score": float(round(ml_score, 6)),
                "risk_score": float(round(risk_score, 6)),
                "risk_band": risk_band,
                "recommended_action": recommended_action,
                "estimated_loss_if_approved": float(round(refund_amount * risk_score, 2)),
                "triggered_rules": triggered_rules,
                "refund_amount": refund_amount,
                "confirmed_abuse_label": int(current_return["confirmed_abuse_label"]),
            }
        )
    return tuple(results)


def score_cases() -> list[dict[str, Any]]:
    """Return cached lightweight scores for every stored return."""
    return [
        {**row, "triggered_rules": list(row["triggered_rules"])}
        for row in _score_cases_cached(_rule_weight())
    ]


def score_case(order_id: str) -> dict[str, Any]:
    """Return the PRD scoring payload for the latest return on an order.

    The final risk score is a configurable weighted blend:
    ``rule_weight * rule_score + (1 - rule_weight) * ml_score``. The default
    is a 50/50 blend; set ``RETURNSHIELD_RULE_WEIGHT`` to any value from 0 to 1
    before calling this function to change the balance.
    """
    context = _get_scoring_context()
    (
        _order,
        current_return,
        _history,
        _rule_score,
        triggered_rules,
        feature_frame,
        _ml_score,
        risk_score,
        risk_band,
        recommended_action,
    ) = _case_components(context, order_id)

    refund_amount = float(current_return.get("refund_amount", 0.0) or 0.0)
    return {
        "risk_score": float(round(risk_score, 6)),
        "risk_band": risk_band,
        "recommended_action": recommended_action,
        "estimated_loss_if_approved": float(round(refund_amount * risk_score, 2)),
        "top_reasons": explain_prediction(context.model, feature_frame),
        "recommended_verification": _recommended_verification(triggered_rules),
    }