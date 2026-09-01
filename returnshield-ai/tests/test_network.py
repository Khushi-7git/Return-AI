"""Tests for linked-account network API endpoints."""

import networkx as nx
import pandas as pd
import pytest
from fastapi import HTTPException

import src.api as api
from src.features import build_linked_account_graph


def _test_graph() -> nx.Graph:
    return build_linked_account_graph(
        pd.DataFrame(
            [
                {
                    "customer_id": "C1",
                    "hashed_address": "address-1",
                    "hashed_device": "device-shared",
                    "hashed_payment": "payment-1",
                },
                {
                    "customer_id": "C2",
                    "hashed_address": "address-2",
                    "hashed_device": "device-shared",
                    "hashed_payment": "payment-2",
                },
                {
                    "customer_id": "C3",
                    "hashed_address": "address-3",
                    "hashed_device": "device-3",
                    "hashed_payment": "payment-3",
                },
            ]
        )
    )


def test_network_endpoints_return_rings_and_linked_history(monkeypatch) -> None:
    monkeypatch.setattr(api, "LINKED_ACCOUNT_GRAPH", _test_graph())
    monkeypatch.setattr(
        api,
        "list_cases",
        lambda: [
            {"customer_id": "C1", "confirmed_abuse_label": 1},
            {"customer_id": "C2", "confirmed_abuse_label": 0},
            {"customer_id": "C2", "confirmed_abuse_label": 1},
        ],
    )

    rings = api.network_rings()
    assert rings == [
        {
            "ring_id": 1,
            "ring_size": 2,
            "customer_ids": ["C1", "C2"],
            "shared_attributes": ["hashed_device"],
        }
    ]

    customer = api.network_customer("C1")
    assert customer == {
        "customer_id": "C1",
        "ring_size": 2,
        "linked_customers": ["C2"],
        "shared_attributes": ["hashed_device"],
        "linked_customer_history": [
            {
                "customer_id": "C2",
                "return_count": 2,
                "confirmed_abuse_labels": [0, 1],
            }
        ],
    }


def test_network_customer_returns_404_for_unknown_customer(monkeypatch) -> None:
    monkeypatch.setattr(api, "LINKED_ACCOUNT_GRAPH", _test_graph())

    with pytest.raises(HTTPException) as error:
        api.network_customer("missing")

    assert error.value.status_code == 404
    assert error.value.detail == "Unknown customer_id in dataset: missing"