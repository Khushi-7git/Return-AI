"""Relationship graph features for linked customer accounts."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
import pandas as pd


LINK_ATTRIBUTES = ("hashed_address", "hashed_device", "hashed_payment")


def _has_value(value: Any) -> bool:
    """Return whether a relationship value is present and non-empty."""
    if value is None or pd.isna(value):
        return False
    return bool(str(value).strip())


def build_linked_account_graph(orders_df: pd.DataFrame) -> nx.Graph:
    """Build a customer graph from shared address, device, or payment values."""
    graph = nx.Graph()

    if "customer_id" not in orders_df.columns:
        raise ValueError("orders_df must contain a customer_id column")

    customer_ids = [
        customer_id
        for customer_id in orders_df["customer_id"].drop_duplicates().tolist()
        if _has_value(customer_id)
    ]
    graph.add_nodes_from(customer_ids)

    for attribute in LINK_ATTRIBUTES:
        if attribute not in orders_df.columns:
            continue
        customers_by_value: dict[str, set[Any]] = {}
        for row in orders_df[["customer_id", attribute]].itertuples(index=False):
            customer_id, shared_value = row
            if not _has_value(customer_id) or not _has_value(shared_value):
                continue
            customers_by_value.setdefault(str(shared_value), set()).add(customer_id)

        for customers in customers_by_value.values():
            for customer_a, customer_b in combinations(
                sorted(customers, key=str),
                2,
            ):
                if graph.has_edge(customer_a, customer_b):
                    shared_via = graph[customer_a][customer_b]["shared_via"]
                    if attribute not in shared_via:
                        shared_via.append(attribute)
                else:
                    graph.add_edge(
                        customer_a,
                        customer_b,
                        shared_via=[attribute],
                    )

    return graph


def get_customer_ring(graph: nx.Graph, customer_id: Any) -> dict[str, Any]:
    """Return the connected customer component and shared link attributes."""
    isolated_result = {
        "customer_id": customer_id,
        "ring_size": 1,
        "linked_customers": [],
        "shared_attributes": [],
    }
    if customer_id not in graph or graph.degree(customer_id) == 0:
        return isolated_result

    component = nx.node_connected_component(graph, customer_id)
    linked_customers = sorted(
        (node for node in component if node != customer_id),
        key=str,
    )
    shared_attributes: set[str] = set()
    for customer_a, customer_b, edge_data in graph.subgraph(component).edges(data=True):
        shared_attributes.update(edge_data.get("shared_via", []))

    return {
        "customer_id": customer_id,
        "ring_size": len(component),
        "linked_customers": linked_customers,
        "shared_attributes": sorted(shared_attributes),
    }