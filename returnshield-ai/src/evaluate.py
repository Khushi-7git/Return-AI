"""Evaluate ReturnShield on the untouched customer test split.

Run with:
    PYTHONPATH=. .venv/bin/python -m src.evaluate

The script fits a fresh Random Forest on train customers only. Test labels are
used only after scoring, for the metrics and retrospective cost comparison.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .cost import configured_costs, policy_statistics
from .model import DATA_DIR, FEATURE_COLUMNS, _build_feature_frame, _rule_weight
from .rules import score_rules


REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "eval_report.md"
CLASSIFICATION_THRESHOLD = 0.50
TOP_REVIEW_FRACTION = 0.05


def _fit_train_only_model(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    train_return_ids: list[str],
) -> RandomForestClassifier:
    labels = returns.set_index("return_id")["confirmed_abuse_label"].astype(int)
    train_features = features.loc[train_return_ids]
    train_labels = labels.loc[train_return_ids]
    if train_labels.nunique() < 2:
        raise ValueError("Train split must contain both abuse-label classes")

    model = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    return model


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders_path = DATA_DIR / "orders.csv"
    returns_path = DATA_DIR / "returns.csv"
    split_path = DATA_DIR / "split.csv"
    if not all(path.exists() for path in (orders_path, returns_path, split_path)):
        raise FileNotFoundError("orders.csv, returns.csv, and split.csv are required")

    orders = pd.read_csv(orders_path)
    returns = pd.read_csv(returns_path)
    split = pd.read_csv(split_path)
    required_split_columns = {"customer_id", "split"}
    if not required_split_columns.issubset(split.columns):
        raise ValueError("split.csv must contain customer_id and split columns")
    return orders, returns, split


def _rule_scores_for_test(
    orders: pd.DataFrame,
    returns_with_customers: pd.DataFrame,
    test_return_ids: set[str],
) -> dict[str, float]:
    order_lookup = orders.set_index("order_id", drop=False)
    test_returns = returns_with_customers.loc[
        returns_with_customers["return_id"].astype(str).isin(test_return_ids)
    ].copy()
    test_returns["timestamp"] = pd.to_datetime(test_returns["timestamp"], errors="coerce")
    test_returns = test_returns.sort_values(
        ["customer_id", "timestamp", "return_id"],
        kind="stable",
    )

    rule_scores: dict[str, float] = {}
    for _customer_id, customer_returns in test_returns.groupby("customer_id", sort=False):
        history: list[dict[str, Any]] = []
        for return_record in customer_returns.to_dict("records"):
            order = order_lookup.loc[return_record["order_id"]]
            rule_score, _triggered_rules = score_rules(order, return_record, history)
            rule_scores[str(return_record["return_id"])] = rule_score
            history.append(return_record)
    return rule_scores


def _positive_probabilities(model: RandomForestClassifier, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
    return probabilities[:, positive_index].astype(float)


def _binary_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, float | int]:
    predictions = (scores >= CLASSIFICATION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _top_capacity_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    review_count = max(1, int(np.ceil(len(frame) * TOP_REVIEW_FRACTION)))
    ranked = frame.sort_values(
        ["risk_score", "return_id"],
        ascending=[False, True],
        kind="stable",
    )
    selected = ranked.head(review_count)
    abuse_count = int(frame["confirmed_abuse_label"].sum())
    selected_abuse = int(selected["confirmed_abuse_label"].sum())
    return {
        "capacity_percent": TOP_REVIEW_FRACTION * 100,
        "review_count": review_count,
        "precision": selected_abuse / review_count if review_count else 0.0,
        "recall": selected_abuse / abuse_count if abuse_count else 0.0,
        "abuse_cases_selected": selected_abuse,
        "abuse_cases_total": abuse_count,
    }


def _calibration_table(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_value, group in frame.groupby(group_column, sort=True, dropna=False):
        predicted = float(group["risk_score"].mean())
        observed = float(group["confirmed_abuse_label"].mean())
        rows.append(
            {
                group_column: str(group_value),
                "cases": int(len(group)),
                "mean_predicted_risk": predicted,
                "observed_abuse_rate": observed,
                "calibration_gap": predicted - observed,
            }
        )
    return rows


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _loss_rows(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = frame.to_dict("records")
    policies = [
        ("Approve-all baseline", None),
        ("Rule-only baseline", "rule_score"),
        ("Blended model", "risk_score"),
    ]
    statistics: dict[str, dict[str, Any]] = {}
    for name, score_key in policies:
        statistics[name] = policy_statistics(cases, score_key)
    approve_all_loss = statistics["Approve-all baseline"]["expected_loss"]
    rule_loss = statistics["Rule-only baseline"]["expected_loss"]
    rows = []
    for name, _score_key in policies:
        loss = statistics[name]["expected_loss"]
        rows.append(
            {
                "policy": name,
                "expected_loss": loss,
                "savings_vs_approve_all": approve_all_loss - loss,
                "savings_vs_rule_only": rule_loss - loss,
                "false_positives": statistics[name]["false_positives"],
                "false_negatives": statistics[name]["false_negatives"],
                "manual_reviews": statistics[name]["manual_reviews"],
            }
        )
    return rows, statistics


def build_report() -> str:
    """Fit on train, evaluate on test, and return the Markdown report."""
    orders, returns, split = _load_splits()
    customer_split = split.set_index("customer_id")["split"]
    joined = returns.merge(
        orders[["order_id", "customer_id", "category", "payment_type"]],
        on="order_id",
        how="inner",
    )
    joined["split"] = joined["customer_id"].map(customer_split)
    if joined["split"].isna().any():
        raise ValueError("Every return must map to a customer split")

    train_customers = set(split.loc[split["split"] == "train", "customer_id"])
    test_customers = set(split.loc[split["split"] == "test", "customer_id"])
    if train_customers & test_customers:
        raise ValueError("Train and test customers overlap")

    train_returns = joined.loc[joined["split"] == "train"]
    test_returns = joined.loc[joined["split"] == "test"].copy()
    if test_returns.empty:
        raise ValueError("Test split is empty")
    train_return_ids = train_returns["return_id"].astype(str).tolist()
    test_return_ids = test_returns["return_id"].astype(str).tolist()

    # Feature construction excludes confirmed_abuse_label. Only train labels
    # enter model.fit; test labels are held back until metric calculation.
    all_features = _build_feature_frame(orders, returns)
    model = _fit_train_only_model(all_features, returns, train_return_ids)
    test_features = all_features.loc[test_return_ids]
    test_returns["ml_score"] = _positive_probabilities(model, test_features)

    returns_with_customers = returns.merge(
        orders[["order_id", "customer_id"]],
        on="order_id",
        how="inner",
    )
    rule_scores = _rule_scores_for_test(
        orders,
        returns_with_customers,
        set(test_return_ids),
    )
    test_returns["rule_score"] = test_returns["return_id"].astype(str).map(rule_scores)
    if test_returns["rule_score"].isna().any():
        raise ValueError("Every test return must have a rule score")

    rule_weight = _rule_weight()
    test_returns["risk_score"] = (
        rule_weight * test_returns["rule_score"]
        + (1 - rule_weight) * test_returns["ml_score"]
    ).clip(0.0, 1.0)
    test_returns["confirmed_abuse_label"] = test_returns["confirmed_abuse_label"].astype(int)

    labels = test_returns["confirmed_abuse_label"]
    overall = _binary_metrics(labels, test_returns["risk_score"])
    top_capacity = _top_capacity_metrics(test_returns)
    loss_rows, loss_statistics = _loss_rows(test_returns)
    costs = configured_costs()
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()

    report = f"""# ReturnShield test-set evaluation

Generated: `{generated_at}`

## Scope and methodology

- Evaluation rows: **{len(test_returns):,}** returns from the untouched `test` customer split.
- Train rows: **{len(train_returns):,}** returns from the `train` customer split.
- Train customers: **{len(train_customers):,}**; test customers: **{len(test_customers):,}**; overlap: **0**.
- The Random Forest was fit only on train rows. `confirmed_abuse_label` was not used as a feature and test labels were used only after scoring.
- The reported model score is the configured rule/ML blend with rule weight **{rule_weight:.2f}**.
- Classification metrics use a score threshold of **{CLASSIFICATION_THRESHOLD:.2f}**. Review policies treat Medium and High bands as manual reviews.

## Overall discrimination

{_markdown_table([overall], ["precision", "recall", "f1", "pr_auc", "false_positive_rate", "true_negatives", "false_positives", "false_negatives", "true_positives"])}

## Top-5% review capacity

The test cases were ranked by blended risk score and only the top **{top_capacity["capacity_percent"]:.0f}%** were selected for review.

{_markdown_table([top_capacity], ["capacity_percent", "review_count", "precision", "recall", "abuse_cases_selected", "abuse_cases_total"])}

## Calibration by category

{_markdown_table(_calibration_table(test_returns, "category"), ["category", "cases", "mean_predicted_risk", "observed_abuse_rate", "calibration_gap"])}

## Calibration by payment type

{_markdown_table(_calibration_table(test_returns, "payment_type"), ["payment_type", "cases", "mean_predicted_risk", "observed_abuse_rate", "calibration_gap"])}

## Expected financial loss

Expected loss uses:

```text
(false_positives * fp_cost) + (false_negatives * fn_cost) +
(manual_reviews * review_cost)
```

Configured costs: FP **{costs["fp_cost"]:.2f}**, FN **{costs["fn_cost"]:.2f}**, review **{costs["review_cost"]:.2f}**, wrong-swap refund **{costs["wrong_swap_refund"]:.2f}**. False-negative wrong-item swaps use the wrong-swap refund instead of the generic FN cost.

{_markdown_table(loss_rows, ["policy", "expected_loss", "savings_vs_approve_all", "savings_vs_rule_only", "false_positives", "false_negatives", "manual_reviews"])}

The blended model's expected loss is **{loss_statistics["Blended model"]["expected_loss"]:.2f}**, versus **{loss_statistics["Approve-all baseline"]["expected_loss"]:.2f}** for approve-all and **{loss_statistics["Rule-only baseline"]["expected_loss"]:.2f}** for rule-only.
"""
    return report


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()