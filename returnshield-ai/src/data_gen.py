"""Deterministic synthetic orders and returns data for ReturnShield AI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ORDERS_COLUMNS = [
    "order_id",
    "customer_id",
    "timestamp",
    "product_id",
    "category",
    "price",
    "payment_type",
    "pincode",
    "hashed_address",
    "hashed_device",
    "hashed_payment",
    "expected_weight_g",
    "shipped_serial",
    "delivery_status",
]

RETURNS_COLUMNS = [
    "return_id",
    "order_id",
    "timestamp",
    "reason",
    "comment",
    "days_after_delivery",
    "pickup_status",
    "refund_amount",
    "received_weight_g",
    "received_serial",
    "packaging_condition",
    "inspection_outcome",
    "confirmed_abuse_label",
]

SPLIT_COLUMNS = ["customer_id", "split"]

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_BASE_DATE = pd.Timestamp("2024-01-01")


def _hash_token(value: str) -> str:
    """Return a stable, non-reversible identifier for a simulated value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build empty frames with the public schemas preserved."""
    return (
        pd.DataFrame(columns=ORDERS_COLUMNS),
        pd.DataFrame(columns=RETURNS_COLUMNS),
        pd.DataFrame(columns=SPLIT_COLUMNS),
    )


def _customer_assignments(
    n_orders: int, n_customers: int, rng: np.random.Generator
) -> np.ndarray:
    """Create at least one order for each customer, then distribute the rest."""
    counts = np.ones(n_customers, dtype=int)
    if n_orders > n_customers:
        counts += rng.multinomial(n_orders - n_customers, np.full(n_customers, 1 / n_customers))

    assignments = np.repeat(np.arange(n_customers), counts)
    rng.shuffle(assignments)
    return assignments


def _make_customer_rings(
    n_customers: int, rng: np.random.Generator
) -> dict[int, int]:
    """Assign a small, disjoint set of customers to coordinated abuse rings."""
    if n_customers < 3:
        return {}

    n_rings = min(12, max(1, n_customers // 40))
    members_per_ring = min(5, max(3, n_customers // n_rings))
    shuffled_customers = rng.permutation(n_customers)
    ring_members: dict[int, int] = {}

    cursor = 0
    for ring_id in range(n_rings):
        members = shuffled_customers[cursor : cursor + members_per_ring]
        cursor += len(members)
        for customer_index in members:
            ring_members[int(customer_index)] = ring_id

    return ring_members


def _split_customers(
    orders: pd.DataFrame,
    n_customers: int,
) -> pd.DataFrame:
    """Assign customers to chronological train, validation, and test groups."""
    first_orders = (
        orders.groupby("customer_id", as_index=False)["timestamp"]
        .min()
        .sort_values(["timestamp", "customer_id"], kind="stable")
        .reset_index(drop=True)
    )

    train_end = max(1, int(n_customers * 0.70))
    validation_end = int(n_customers * 0.85)
    if n_customers >= 3:
        train_end = min(train_end, n_customers - 2)
        validation_end = min(n_customers - 1, max(train_end + 1, validation_end))
    else:
        validation_end = max(train_end, min(n_customers, validation_end))

    splits = np.full(n_customers, "test", dtype=object)
    splits[:train_end] = "train"
    splits[train_end:validation_end] = "validation"

    split_by_customer = {
        customer_id: splits[chronological_position]
        for chronological_position, customer_id in enumerate(first_orders["customer_id"])
    }
    return pd.DataFrame(
        {
            "customer_id": first_orders["customer_id"],
            "split": [split_by_customer[customer_id] for customer_id in first_orders["customer_id"]],
        }
    )[SPLIT_COLUMNS]


def _return_type_for_claim(
    claim_type: str,
    index: int,
) -> tuple[str, str]:
    """Map a simulated return scenario to a reason and customer-facing comment."""
    if claim_type == "product_swap":
        return (
            "wrong_item_received",
            "The returned item does not match the product that was shipped.",
        )
    if claim_type == "false_defect":
        return (
            "defective",
            "The customer reported that the item was defective.",
        )
    if claim_type == "ring_abuse":
        if index % 2:
            return (
                "defective",
                "The item was reported as defective by a customer in a coordinated pattern.",
            )
        return (
            "wrong_item_received",
            "The return appears to be part of a coordinated item-swap pattern.",
        )

    legitimate_reasons = [
        ("changed_mind", "The customer changed their mind about the purchase."),
        ("size_issue", "The customer reported that the size was not suitable."),
        ("arrived_late", "The customer reported that the order arrived later than expected."),
    ]
    return legitimate_reasons[index % len(legitimate_reasons)]


def generate_dataset(
    n_orders: int = 20000,
    n_returns: int = 3000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and save deterministic synthetic orders, returns, and splits.

    The returned tuple is ``(orders, returns)``. The customer split is saved
    separately as ``data/split.csv``. The abuse ground-truth label is included
    only in the returns frame and is never used to construct an order feature.
    """
    if not isinstance(n_orders, (int, np.integer)) or not isinstance(
        n_returns, (int, np.integer)
    ):
        raise TypeError("n_orders and n_returns must be integers")
    if n_orders < 0 or n_returns < 0:
        raise ValueError("n_orders and n_returns must be non-negative")
    if n_returns > n_orders:
        raise ValueError("n_returns cannot exceed n_orders")

    if n_orders == 0:
        orders, returns, split = _empty_frames()
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        orders.to_csv(_DATA_DIR / "orders.csv", index=False)
        returns.to_csv(_DATA_DIR / "returns.csv", index=False)
        split.to_csv(_DATA_DIR / "split.csv", index=False)
        return orders, returns

    rng = np.random.default_rng(seed)
    n_customers = min(n_orders, max(10, n_orders // 5))
    customer_assignments = _customer_assignments(n_orders, n_customers, rng)
    first_order_dates = _BASE_DATE + pd.to_timedelta(
        rng.integers(0, 730, size=n_customers), unit="D"
    )

    seen_customers = np.zeros(n_customers, dtype=bool)
    order_offsets = rng.integers(1, 366, size=n_orders)
    for row_index, customer_index in enumerate(customer_assignments):
        if not seen_customers[customer_index]:
            order_offsets[row_index] = 0
            seen_customers[customer_index] = True
    order_timestamps = first_order_dates[customer_assignments] + pd.to_timedelta(
        order_offsets, unit="D"
    )

    categories = np.array(["electronics", "home", "apparel", "beauty", "sports"])
    catalog_size = max(20, min(500, n_orders // 3 + 1))
    catalog_categories = rng.choice(categories, size=catalog_size)
    catalog_prices = np.round(
        np.clip(rng.lognormal(mean=4.2, sigma=0.65, size=catalog_size), 8, 1800), 2
    )
    catalog_weights = np.round(
        np.clip(rng.lognormal(mean=6.6, sigma=0.55, size=catalog_size), 80, 12000), 1
    )
    product_indices = rng.integers(0, catalog_size, size=n_orders)
    product_ids = np.array([f"SKU-{index:05d}" for index in product_indices])

    ring_members = _make_customer_rings(n_customers, rng)
    ring_by_customer = np.array(
        [ring_members.get(customer_index, -1) for customer_index in range(n_customers)]
    )
    customer_ring_ids = ring_by_customer[customer_assignments]

    hashed_addresses: list[str] = []
    hashed_devices: list[str] = []
    hashed_payments: list[str] = []
    for customer_index, ring_id in zip(customer_assignments, customer_ring_ids):
        if ring_id >= 0:
            hashed_addresses.append(_hash_token(f"ring-address-{ring_id}"))
            hashed_devices.append(_hash_token(f"ring-device-{ring_id}"))
            hashed_payments.append(_hash_token(f"ring-payment-{ring_id}"))
        else:
            hashed_addresses.append(_hash_token(f"address-{customer_index}"))
            hashed_devices.append(_hash_token(f"device-{customer_index}"))
            hashed_payments.append(_hash_token(f"payment-{customer_index}"))

    orders = pd.DataFrame(
        {
            "order_id": [f"ORD-{index:07d}" for index in range(1, n_orders + 1)],
            "customer_id": [
                f"CUST-{customer_index + 1:06d}" for customer_index in customer_assignments
            ],
            "timestamp": order_timestamps,
            "product_id": product_ids,
            "category": catalog_categories[product_indices],
            "price": catalog_prices[product_indices],
            "payment_type": rng.choice(
                ["card", "upi", "wallet", "cash_on_delivery"],
                size=n_orders,
                p=[0.48, 0.28, 0.14, 0.10],
            ),
            "pincode": [f"{int(value):06d}" for value in rng.integers(100000, 999999, n_orders)],
            "hashed_address": hashed_addresses,
            "hashed_device": hashed_devices,
            "hashed_payment": hashed_payments,
            "expected_weight_g": catalog_weights[product_indices],
            "shipped_serial": [
                _hash_token(f"serial-{index}-{product_id}")
                for index, product_id in enumerate(product_ids, start=1)
            ],
            "delivery_status": "delivered",
        },
        columns=ORDERS_COLUMNS,
    )

    if n_returns == 0:
        returns = pd.DataFrame(columns=RETURNS_COLUMNS)
    else:
        ring_order_indices = np.flatnonzero(customer_ring_ids >= 0)
        ring_return_count = min(
            len(ring_order_indices),
            max(1, int(round(n_returns * 0.04))),
        )
        ring_selected = (
            rng.choice(ring_order_indices, size=ring_return_count, replace=False)
            if ring_return_count
            else np.array([], dtype=int)
        )
        remaining_indices = np.setdiff1d(np.arange(n_orders), ring_selected, assume_unique=False)
        regular_return_count = n_returns - len(ring_selected)
        regular_selected = (
            rng.choice(remaining_indices, size=regular_return_count, replace=False)
            if regular_return_count
            else np.array([], dtype=int)
        )
        selected_order_indices = np.concatenate([ring_selected, regular_selected])
        rng.shuffle(selected_order_indices)

        claim_types = np.full(n_returns, "legitimate", dtype=object)
        ring_lookup = {order_index: "ring_abuse" for order_index in ring_selected}
        for return_index, order_index in enumerate(selected_order_indices):
            if order_index in ring_lookup:
                claim_types[return_index] = ring_lookup[order_index]
            else:
                claim_types[return_index] = rng.choice(
                    ["legitimate", "product_swap", "false_defect"],
                    p=[0.76, 0.12, 0.12],
                )

        # Keep the small-sample generator representative as well as random.
        regular_positions = [
            index for index, claim_type in enumerate(claim_types) if claim_type == "legitimate"
        ]
        if len(regular_positions) >= 3:
            claim_types[regular_positions[0:3]] = [
                "legitimate",
                "product_swap",
                "false_defect",
            ]

        selected_orders = orders.iloc[selected_order_indices].reset_index(drop=True)
        return_timestamps: list[pd.Timestamp] = []
        reasons: list[str] = []
        comments: list[str] = []
        days_after_delivery: list[int] = []
        pickup_statuses: list[str] = []
        refund_amounts: list[float] = []
        received_weights: list[float] = []
        received_serials: list[str] = []
        packaging_conditions: list[str] = []
        inspection_outcomes: list[str] = []
        labels: list[int] = []

        for return_index, (claim_type, order) in enumerate(
            zip(claim_types, selected_orders.to_dict("records"))
        ):
            reason, comment = _return_type_for_claim(str(claim_type), return_index)
            days_since_delivery = int(rng.integers(1, 31))
            delivery_lag = int(rng.integers(2, 8))
            return_timestamps.append(
                pd.Timestamp(order["timestamp"])
                + pd.Timedelta(days=delivery_lag + days_since_delivery)
            )
            reasons.append(reason)
            comments.append(comment)
            days_after_delivery.append(days_since_delivery)
            pickup_statuses.append(
                str(rng.choice(["picked_up", "pickup_pending"], p=[0.94, 0.06]))
            )

            is_abuse = claim_type in {"product_swap", "false_defect", "ring_abuse"}
            labels.append(int(is_abuse))
            refund_amounts.append(
                round(float(order["price"]) * float(rng.uniform(0.92, 1.0)), 2)
            )

            if claim_type in {"product_swap", "ring_abuse"}:
                received_weights.append(
                    round(float(order["expected_weight_g"]) * float(rng.uniform(0.35, 0.72)), 1)
                )
                received_serials.append(_hash_token(f"replacement-serial-{return_index}"))
                packaging_conditions.append("tampered")
                inspection_outcomes.append("serial_mismatch")
            elif claim_type == "false_defect":
                received_weights.append(
                    round(float(order["expected_weight_g"]) * float(rng.uniform(0.97, 1.03)), 1)
                )
                received_serials.append(str(order["shipped_serial"]))
                packaging_conditions.append(str(rng.choice(["intact", "opened"])))
                inspection_outcomes.append("no_issue_found")
            else:
                received_weights.append(
                    round(float(order["expected_weight_g"]) * float(rng.uniform(0.94, 1.06)), 1)
                )
                received_serials.append(str(order["shipped_serial"]))
                packaging_conditions.append(str(rng.choice(["intact", "opened", "damaged"])))
                inspection_outcomes.append("verified")

        returns = pd.DataFrame(
            {
                "return_id": [f"RET-{index:07d}" for index in range(1, n_returns + 1)],
                "order_id": selected_orders["order_id"].tolist(),
                "timestamp": return_timestamps,
                "reason": reasons,
                "comment": comments,
                "days_after_delivery": days_after_delivery,
                "pickup_status": pickup_statuses,
                "refund_amount": refund_amounts,
                "received_weight_g": received_weights,
                "received_serial": received_serials,
                "packaging_condition": packaging_conditions,
                "inspection_outcome": inspection_outcomes,
                "confirmed_abuse_label": labels,
            },
            columns=RETURNS_COLUMNS,
        )

    split = _split_customers(orders, n_customers)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_csv(_DATA_DIR / "orders.csv", index=False)
    returns.to_csv(_DATA_DIR / "returns.csv", index=False)
    split.to_csv(_DATA_DIR / "split.csv", index=False)
    return orders, returns