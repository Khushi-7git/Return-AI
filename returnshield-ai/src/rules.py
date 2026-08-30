"""Explainable rule-based baseline for return-risk scoring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import pandas as pd


def _value(record: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping, Series, or lightweight object."""
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(key, default)
    if isinstance(record, pd.Series):
        return record.get(key, default)
    return getattr(record, key, default)


def _history_records(history: Any) -> list[Any]:
    """Normalize common history inputs into a list of return-like records."""
    if history is None:
        return []
    if isinstance(history, pd.DataFrame):
        return history.to_dict("records")
    if isinstance(history, Mapping):
        nested = history.get("returns")
        if nested is not None:
            return _history_records(nested)
        return [history]
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


def score_rules(
    order: Any,
    return_record: Any,
    history: Any,
    weight_mismatch_threshold: float = 0.20,
) -> tuple[float, list[str]]:
    """Score a return using explainable risk rules.

    ``history`` may be a list of mappings/Series or a returns DataFrame. It is
    expected to contain returns for the same customer, generally excluding the
    current return. The confirmed abuse label is intentionally never read.

    Args:
        order: Order-like mapping or Series.
        return_record: Current return-like mapping or Series.
        history: Earlier return records for the customer.
        weight_mismatch_threshold: Relative weight difference that triggers
            ``weight_mismatch``. For example, ``0.20`` means 20 percent.

    Returns:
        A ``(score, triggered_rules)`` tuple where score is clipped to 0-1.
    """
    if not 0 <= weight_mismatch_threshold:
        raise ValueError("weight_mismatch_threshold must be non-negative")

    records = _history_records(history)
    current_timestamp = _timestamp(return_record)
    current_return_id = _value(return_record, "return_id")

    prior_records: list[Any] = []
    for record in records:
        if current_return_id is not None and _value(record, "return_id") == current_return_id:
            continue
        record_timestamp = _timestamp(record)
        if (
            current_timestamp is not None
            and record_timestamp is not None
            and record_timestamp > current_timestamp
        ):
            continue
        prior_records.append(record)

    recent_prior_returns = [
        record
        for record in prior_records
        if current_timestamp is None
        or _timestamp(record) is None
        or (current_timestamp - _timestamp(record)).total_seconds() <= 90 * 86400
    ]

    triggered_rules: list[str] = []
    score = 0.0

    if recent_prior_returns:
        triggered_rules.append("repeated_returns_90d")
        score += 0.25

    expected_weight = _value(order, "expected_weight_g")
    received_weight = _value(return_record, "received_weight_g")
    if expected_weight is not None and received_weight is not None:
        try:
            expected = float(expected_weight)
            received = float(received_weight)
        except (TypeError, ValueError):
            expected = received = 0.0
        if expected > 0 and abs(expected - received) / expected > weight_mismatch_threshold:
            triggered_rules.append("weight_mismatch")
            score += 0.30

    shipped_serial = _value(order, "shipped_serial")
    received_serial = _value(return_record, "received_serial")
    if shipped_serial not in (None, "") and received_serial not in (None, ""):
        if str(shipped_serial) != str(received_serial):
            triggered_rules.append("serial_mismatch")
            score += 0.35

    order_sku = _value(order, "product_id", _value(order, "sku"))
    return_sku = _value(return_record, "product_id", _value(return_record, "sku"))
    if order_sku not in (None, "") and return_sku not in (None, ""):
        if str(order_sku) != str(return_sku):
            triggered_rules.append("sku_mismatch")
            score += 0.35

    current_family = _claim_family(_value(return_record, "reason"))
    prior_families = [_claim_family(_value(record, "reason")) for record in prior_records]
    if current_family == "false_defect" and "false_defect" in prior_families:
        triggered_rules.append("repeated_false_defect_claims")
        score += 0.25
    if current_family == "swap" and "swap" in prior_families:
        triggered_rules.append("repeated_swap_claims")
        score += 0.25

    return min(1.0, round(score, 6)), triggered_rules