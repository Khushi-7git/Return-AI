"""Tests for linked customer account graph features."""

import networkx as nx
import pandas as pd

from src.features import build_linked_account_graph, get_customer_ring


def test_shared_device_links_customers_but_not_unrelated_customer() -> None:
    orders = pd.DataFrame(
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
                "hashed_address": None,
                "hashed_device": "",
                "hashed_payment": "payment-3",
            },
        ]
    )

    graph = build_linked_account_graph(orders)

    assert isinstance(graph, nx.Graph)
    assert set(nx.node_connected_component(graph, "C1")) == {"C1", "C2"}
    assert set(nx.node_connected_component(graph, "C2")) == {"C1", "C2"}
    assert set(nx.node_connected_component(graph, "C3")) == {"C3"}
    assert graph["C1"]["C2"]["shared_via"] == ["hashed_device"]

    assert get_customer_ring(graph, "C1") == {
        "customer_id": "C1",
        "ring_size": 2,
        "linked_customers": ["C2"],
        "shared_attributes": ["hashed_device"],
    }
    assert get_customer_ring(graph, "missing") == {
        "customer_id": "missing",
        "ring_size": 1,
        "linked_customers": [],
        "shared_attributes": [],
    }