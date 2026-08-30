"""Tests for deterministic data generation and customer-separated splits."""

from pathlib import Path

import pandas as pd

from src.data_gen import generate_dataset


def test_customer_ids_are_exclusive_to_one_split(tmp_path: Path, monkeypatch) -> None:
    import src.data_gen as data_gen

    monkeypatch.setattr(data_gen, "_DATA_DIR", tmp_path)
    _, _ = generate_dataset(n_orders=120, n_returns=24, seed=42)
    split = pd.read_csv(tmp_path / "split.csv")

    assignments_per_customer = split.groupby("customer_id")["split"].nunique()

    assert assignments_per_customer.max() == 1
    assert set(split["split"]) == {"train", "validation", "test"}