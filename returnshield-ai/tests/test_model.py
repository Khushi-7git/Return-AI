"""End-to-end checks for the model scoring payload."""

import pandas as pd

from src.model import FEATURE_COLUMNS, score_case


def test_score_case_matches_prd_output_shape() -> None:
    returns = pd.read_csv("data/returns.csv")
    order_id = str(returns.iloc[0]["order_id"])

    result = score_case(order_id)

    assert set(result) == {
        "risk_score",
        "risk_band",
        "recommended_action",
        "estimated_loss_if_approved",
        "top_reasons",
        "recommended_verification",
    }
    assert isinstance(result["risk_score"], float)
    assert 0 <= result["risk_score"] <= 1
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert result["recommended_action"] in {"approve", "verify", "manual_review"}
    assert isinstance(result["estimated_loss_if_approved"], float)
    assert 3 <= len(result["top_reasons"]) <= 5
    assert all(isinstance(reason, str) and reason for reason in result["top_reasons"])
    assert isinstance(result["recommended_verification"], str)
    assert "confirmed_abuse_label" not in FEATURE_COLUMNS