"""Plain-English SHAP explanations for return-risk predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import shap


def _as_feature_frame(feature_row: Any, feature_names: list[str] | None) -> pd.DataFrame:
    """Normalize a feature row and align it with the model's feature order."""
    if isinstance(feature_row, pd.DataFrame):
        frame = feature_row.copy().iloc[:1]
    elif isinstance(feature_row, pd.Series):
        frame = pd.DataFrame([feature_row.to_dict()])
    elif isinstance(feature_row, Mapping):
        frame = pd.DataFrame([dict(feature_row)])
    else:
        raise TypeError("feature_row must be a mapping, pandas Series, or DataFrame")

    if frame.empty:
        raise ValueError("feature_row cannot be empty")

    if feature_names is not None:
        for name in feature_names:
            if name not in frame:
                frame[name] = 0.0
        frame = frame[feature_names]
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _shap_values_for_positive_class(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Calculate one-dimensional SHAP values for the positive class."""
    try:
        explanation = shap.TreeExplainer(model)(frame)
    except Exception as tree_error:
        try:
            explanation = shap.Explainer(model, frame)(frame)
        except Exception as explainer_error:
            raise RuntimeError("Unable to calculate SHAP values for this model") from explainer_error

    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # Tree SHAP commonly returns (rows, features, classes).
        class_index = 1 if values.shape[-1] > 1 else 0
        values = values[0, :, class_index]
    elif values.ndim == 2:
        values = values[0]
    elif values.ndim != 1:
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")
    return values.astype(float)


def _number(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _describe_feature(feature_name: str, row: pd.Series) -> str:
    """Translate a model feature into a user-facing explanation."""
    if feature_name == "weight_difference_g":
        difference = _number(row, feature_name)
        if difference > 0:
            return f"returned item weight is {difference:.0f}g lighter than expected"
        if difference < 0:
            return f"returned item weight is {abs(difference):.0f}g heavier than expected"
        return "returned item weight matches the expected weight"

    if feature_name == "weight_mismatch_ratio":
        return (
            f"returned item weight differs from expected by "
            f"{abs(_number(row, feature_name)) * 100:.1f}%"
        )
    if feature_name == "serial_mismatch":
        return (
            "returned item serial does not match the shipped serial"
            if _number(row, feature_name) > 0
            else "returned item serial matches the shipped serial"
        )
    if feature_name == "sku_mismatch":
        return (
            "returned product SKU does not match the order SKU"
            if _number(row, feature_name) > 0
            else "returned product SKU matches the order SKU"
        )
    if feature_name == "days_after_delivery":
        return f"return was requested {_number(row, feature_name):.0f} days after delivery"
    if feature_name == "refund_ratio":
        return f"refund amount is {_number(row, feature_name) * 100:.0f}% of the order price"
    if feature_name == "repeat_returns_90d":
        count = _number(row, feature_name)
        return f"customer has {count:.0f} other return(s) in the last 90 days"
    if feature_name == "customer_return_count":
        return f"customer has {(_number(row, feature_name)):.0f} prior return(s)"
    if feature_name == "prior_false_defect_claims":
        return f"customer has {(_number(row, feature_name)):.0f} prior defect claim(s)"
    if feature_name == "prior_swap_claims":
        return f"customer has {(_number(row, feature_name)):.0f} prior wrong-item claim(s)"
    if feature_name == "address_customer_count":
        return (
            f"the same hashed address is associated with "
            f"{_number(row, feature_name):.0f} customer account(s)"
        )
    if feature_name == "device_customer_count":
        return (
            f"the same hashed device is associated with "
            f"{_number(row, feature_name):.0f} customer account(s)"
        )
    if feature_name == "claim_is_defect":
        return (
            "return reason is a defect claim"
            if _number(row, feature_name) > 0
            else "return reason is not a defect claim"
        )
    if feature_name == "claim_is_swap":
        return (
            "return reason is a wrong-item claim"
            if _number(row, feature_name) > 0
            else "return reason is not a wrong-item claim"
        )
    if feature_name == "packaging_tampered":
        return (
            "returned packaging was marked tampered"
            if _number(row, feature_name) > 0
            else "returned packaging was not marked tampered"
        )
    if feature_name == "expected_weight_g":
        return f"expected item weight is {_number(row, feature_name):.0f}g"
    if feature_name == "received_weight_g":
        return f"received item weight is {_number(row, feature_name):.0f}g"
    if feature_name == "price":
        return f"order value is {_number(row, feature_name):.2f}"
    return f"{feature_name.replace('_', ' ')} is {_number(row, feature_name):.2f}"


def explain_prediction(model: Any, feature_row: Any) -> list[str]:
    """Return the five most influential SHAP reasons in plain English.

    Positive SHAP contributions are described as increasing predicted abuse
    risk; negative contributions are described as lowering it. The helper
    accepts a mapping, Series, or one-row DataFrame and uses
    ``model.feature_names_in_`` when available to preserve feature order.
    """
    model_feature_names = getattr(model, "feature_names_in_", None)
    feature_names = list(model_feature_names) if model_feature_names is not None else None
    frame = _as_feature_frame(feature_row, feature_names)
    if feature_names is None:
        feature_names = list(frame.columns)

    shap_values = _shap_values_for_positive_class(model, frame)
    if len(shap_values) != len(feature_names):
        raise RuntimeError("SHAP values do not match the model feature count")

    row = frame.iloc[0]
    ranked_indices = np.argsort(-np.abs(shap_values))
    target_count = min(5, max(3, len(feature_names)))
    reasons: list[str] = []
    for index in ranked_indices:
        description = _describe_feature(feature_names[index], row)
        direction = "increases" if shap_values[index] >= 0 else "lowers"
        reason = f"{description}; this {direction} predicted abuse risk"
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= target_count:
            break
    return reasons