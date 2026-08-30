"""Configurable financial loss calculations for ReturnShield policies."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_FP_COST = 150.0
DEFAULT_FN_COST = 1200.0
DEFAULT_REVIEW_COST = 40.0
DEFAULT_WRONG_SWAP_REFUND = 1500.0


def _configured_cost(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative number") from error
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def configured_costs() -> dict[str, float]:
    """Return cost assumptions, allowing deployment-time environment overrides."""
    return {
        "fp_cost": _configured_cost("RETURNSHIELD_FP_COST", DEFAULT_FP_COST),
        "fn_cost": _configured_cost("RETURNSHIELD_FN_COST", DEFAULT_FN_COST),
        "review_cost": _configured_cost("RETURNSHIELD_REVIEW_COST", DEFAULT_REVIEW_COST),
        "wrong_swap_refund": _configured_cost(
            "RETURNSHIELD_WRONG_SWAP_REFUND",
            DEFAULT_WRONG_SWAP_REFUND,
        ),
    }


def expected_loss(
    false_positives: int | float,
    false_negatives: int | float,
    manual_reviews: int | float,
    fp_cost: float | None = None,
    fn_cost: float | None = None,
    review_cost: float | None = None,
    wrong_swap_refund: float | None = None,
) -> float:
    """Calculate expected loss from classification and review counts.

    The core policy is intentionally explicit:
    ``(false_positives * fp_cost) + (false_negatives * fn_cost) +
    (manual_reviews * review_cost)``.

    ``wrong_swap_refund`` is accepted as part of the configurable cost surface
    and is applied by :func:`policy_statistics` to false-negative wrong-item
    swaps. It is not added to this count-only formula.
    """
    configured = configured_costs()
    fp_cost = configured["fp_cost"] if fp_cost is None else float(fp_cost)
    fn_cost = configured["fn_cost"] if fn_cost is None else float(fn_cost)
    review_cost = configured["review_cost"] if review_cost is None else float(review_cost)
    # Validate the configured fourth cost even though the requested formula is
    # count-based and uses the generic FN cost.
    if wrong_swap_refund is None:
        wrong_swap_refund = configured["wrong_swap_refund"]
    float(wrong_swap_refund)
    return round(
        float(false_positives) * fp_cost
        + float(false_negatives) * fn_cost
        + float(manual_reviews) * review_cost,
        2,
    )


def policy_action(score: float) -> str:
    """Map a 0-1 score to the approve/verify/manual-review action bands."""
    if score < 0.33:
        return "approve"
    if score < 0.66:
        return "verify"
    return "manual_review"


def policy_statistics(
    cases: Iterable[Mapping[str, Any]],
    score_key: str | None,
) -> dict[str, Any]:
    """Calculate count-based loss statistics for one review policy.

    ``score_key=None`` is the approve-all baseline. For scored policies,
    verify and manual-review actions are treated as manual reviews; this
    matches the operational action bands used by the application.
    """
    materialized_cases = list(cases)
    predictions = [
        False
        if score_key is None
        else policy_action(float(case[score_key])) != "approve"
        for case in materialized_cases
    ]
    actual = [bool(int(case.get("confirmed_abuse_label", 0) or 0)) for case in materialized_cases]
    false_positives = sum(predicted and not abused for predicted, abused in zip(predictions, actual))
    false_negatives = sum(not predicted and abused for predicted, abused in zip(predictions, actual))
    manual_reviews = sum(predictions)
    wrong_swap_false_negatives = sum(
        not predicted
        and abused
        and str(case.get("reason", "")) == "wrong_item_received"
        for case, predicted, abused in zip(materialized_cases, predictions, actual)
    )

    costs = configured_costs()
    loss = expected_loss(
        false_positives,
        false_negatives,
        manual_reviews,
        **costs,
    )
    loss += wrong_swap_false_negatives * (costs["wrong_swap_refund"] - costs["fn_cost"])
    action_counts = {"approve": 0, "verify": 0, "manual_review": 0}
    for case, predicted in zip(materialized_cases, predictions):
        action = "approve" if not predicted else policy_action(float(case[score_key]))
        action_counts[action] += 1
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "manual_reviews": manual_reviews,
        "wrong_swap_false_negatives": wrong_swap_false_negatives,
        "expected_loss": round(loss, 2),
        "action_counts": action_counts,
    }


def policy_loss(
    cases: Iterable[Mapping[str, Any]],
    score_key: str | None,
) -> tuple[float, dict[str, int]]:
    """Compatibility wrapper returning loss and action counts."""
    statistics = policy_statistics(cases, score_key)
    return statistics["expected_loss"], statistics["action_counts"]


def financial_impact(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare approve-all, rule-based, and blended-model policies."""
    materialized_cases = list(cases)
    baseline = policy_statistics(materialized_cases, None)
    rule_based = policy_statistics(materialized_cases, "rule_score")
    model_based = policy_statistics(materialized_cases, "risk_score")
    return {
        "baseline_approve_all_loss": baseline["expected_loss"],
        "rule_based_policy_loss": rule_based["expected_loss"],
        "model_policy_loss": model_based["expected_loss"],
        "savings_vs_approve_all": {
            "rule_based": round(
                baseline["expected_loss"] - rule_based["expected_loss"],
                2,
            ),
            "model": round(
                baseline["expected_loss"] - model_based["expected_loss"],
                2,
            ),
        },
        "policy_counts": {
            "approve_all": baseline["action_counts"],
            "rule_based": rule_based["action_counts"],
            "model": model_based["action_counts"],
        },
        "policy_statistics": {
            "approve_all": baseline,
            "rule_based": rule_based,
            "model": model_based,
        },
        "assumptions": configured_costs(),
    }