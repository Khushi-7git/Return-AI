"""Smoke tests for the project skeleton and API metric shape."""

import src.api as api


def test_hello_world_endpoint() -> None:
    assert api.app.title == "ReturnShield AI"
    assert api.hello_world() == {"message": "Hello from ReturnShield AI"}


def test_performance_reports_top_capacity_metrics(monkeypatch) -> None:
    cases = [
        {
            "risk_score": 0.95,
            "confirmed_abuse_label": 1,
            "risk_band": "High",
            "category": "apparel",
            "payment_type": "card",
        },
        {
            "risk_score": 0.80,
            "confirmed_abuse_label": 0,
            "risk_band": "High",
            "category": "apparel",
            "payment_type": "card",
        },
        {
            "risk_score": 0.65,
            "confirmed_abuse_label": 1,
            "risk_band": "Medium",
            "category": "beauty",
            "payment_type": "upi",
        },
        {
            "risk_score": 0.20,
            "confirmed_abuse_label": 0,
            "risk_band": "Low",
            "category": "beauty",
            "payment_type": "upi",
        },
    ]
    monkeypatch.setattr(api, "score_cases", lambda: cases)

    metrics = api.performance()

    assert metrics["top_capacity_metrics"] == {
        "capacity_pct": 5,
        "cases_reviewed": 1,
        "precision": 1.0,
        "recall": 0.5,
    }
    assert metrics["overall"]["false_positive_rate"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["by_category"]["apparel"] == {
        "count": 2,
        "abuse_rate": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
        "pr_auc": 1.0,
    }
    assert metrics["by_payment_type"]["upi"]["count"] == 2