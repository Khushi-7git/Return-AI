"""Financial impact calculations for return approval policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


VERIFICATION_COST = 2.50
MANUAL_REVIEW_COST = 7.50
VERIFICATION_CAPTURE_RATE = 0.70
MANUAL_REVIEW_CAPTURE_RATE = 0.95


def policy_action(score: float) -> str:
    """Map a 0-1 policy score to the same action bands used by the model."""
    if score < 0.33:
        return "approve"
    if score < 0.66:
        return "verify"
    return "manual_review"


def _case_loss(case: Mapping[str, Any], action: str) -> float:
    refund_amount = float(case.get("refund_amount", 0.0) or 0.0)
    abuse_label = int(case.get("confirmed_abuse_label", 0) or 0)
    if action == "approve":
        return refund_amount if abuse_label else 0.0
    if action == "verify":
        residual_abuse_loss = refund_amount * (1 - VERIFICATION_CAPTURE_RATE)
        return residual_abuse_loss if abuse_label else VERIFICATION_COST
    if action == "manual_review":
        residual_abuse_loss = refund_amount * (1 - MANUAL_REVIEW_CAPTURE_RATE)
        return residual_abuse_loss if abuse_label else MANUAL_REVIEW_COST
    raise ValueError(f"Unknown policy action: {action}")


def policy_loss(
    cases: Iterable[Mapping[str, Any]],
    score_key: str | None,
) -> tuple[float, dict[str, int]]:
    """Calculate policy loss and action counts for a collection of cases.

    ``score_key=None`` represents approve-all. Otherwise the named case score
    is converted to approve/verify/manual_review using the shared thresholds.
    Verification and manual review include their operating cost and leave a
    residual abuse loss based on their capture-rate assumptions.
    """
    total_loss = 0.0
    action_counts = {"approve": 0, "verify": 0, "manual_review": 0}
    for case in cases:
        action = "approve" if score_key is None else policy_action(float(case[score_key]))
        action_counts[action] += 1
        total_loss += _case_loss(case, action)
    return round(total_loss, 2), action_counts


def financial_impact(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare approve-all, rule-based, and blended-model policies."""
    materialized_cases = list(cases)
    baseline_loss, baseline_counts = policy_loss(materialized_cases, None)
    rule_loss, rule_counts = policy_loss(materialized_cases, "rule_score")
    model_loss, model_counts = policy_loss(materialized_cases, "risk_score")
    return {
        "baseline_approve_all_loss": baseline_loss,
        "rule_based_policy_loss": rule_loss,
        "model_policy_loss": model_loss,
        "savings_vs_approve_all": {
            "rule_based": round(baseline_loss - rule_loss, 2),
            "model": round(baseline_loss - model_loss, 2),
        },
        "policy_counts": {
            "approve_all": baseline_counts,
            "rule_based": rule_counts,
            "model": model_counts,
        },
        "assumptions": {
            "verification_cost": VERIFICATION_COST,
            "manual_review_cost": MANUAL_REVIEW_COST,
            "verification_capture_rate": VERIFICATION_CAPTURE_RATE,
            "manual_review_capture_rate": MANUAL_REVIEW_CAPTURE_RATE,
        },
    }