"""Tests for the explainable rule baseline."""

from src.rules import score_rules


def test_rules_flag_weight_serial_and_repeat_claims() -> None:
    order = {
        "product_id": "SKU-1",
        "expected_weight_g": 1000,
        "shipped_serial": "SERIAL-1",
    }
    current_return = {
        "return_id": "RET-2",
        "timestamp": "2024-04-01",
        "reason": "defective",
        "received_weight_g": 600,
        "received_serial": "SERIAL-2",
    }
    history = [
        {
            "return_id": "RET-1",
            "timestamp": "2024-03-15",
            "reason": "defective",
        }
    ]

    score, triggered_rules = score_rules(order, current_return, history)

    assert 0 < score <= 1
    assert {
        "repeated_returns_90d",
        "weight_mismatch",
        "serial_mismatch",
        "repeated_false_defect_claims",
    }.issubset(triggered_rules)


def test_rules_do_not_use_ground_truth_label() -> None:
    order = {
        "product_id": "SKU-1",
        "expected_weight_g": 1000,
        "shipped_serial": "SERIAL-1",
    }
    clean_return = {
        "return_id": "RET-1",
        "timestamp": "2024-04-01",
        "reason": "changed_mind",
        "received_weight_g": 1000,
        "received_serial": "SERIAL-1",
        "confirmed_abuse_label": 1,
    }

    score, triggered_rules = score_rules(order, clean_return, [])

    assert score == 0
    assert triggered_rules == []