"""Streamlit dashboard backed entirely by the ReturnShield FastAPI service."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import matplotlib
import networkx as nx
import pandas as pd
import requests
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


API_BASE_URL = os.getenv("RETURNSHIELD_API_URL", "http://127.0.0.1:8000").rstrip("/")
NETWORK_ATTRIBUTE_LABELS = {
    "hashed_address": "address",
    "hashed_device": "device",
    "hashed_payment": "payment",
}
RISK_BAND_STYLES = {
    "Low": "background-color: #e8f3e8; color: #245b2a;",
    "Medium": "background-color: #f8efd0; color: #765c00;",
    "High": "background-color: #f7dddd; color: #8a2f2f;",
}

st.set_page_config(page_title="ReturnShield AI", page_icon="R", layout="wide")


def api_get(path: str) -> Any:
    """Fetch JSON from the FastAPI service and show a useful UI error."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(f"FastAPI service unavailable: {error}")
        st.stop()


def api_post(path: str, **kwargs: Any) -> Any:
    """Post to the FastAPI service and show a useful UI error."""
    try:
        response = requests.post(f"{API_BASE_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(f"FastAPI request failed: {error}")
        st.stop()


@st.cache_data(ttl=30)
def load_queue() -> list[dict[str, Any]]:
    return api_get("/queue")


@st.cache_data(ttl=30)
def load_performance() -> dict[str, Any]:
    return api_get("/performance")


@st.cache_data(ttl=30)
def load_financial() -> dict[str, Any]:
    return api_get("/financial")


@st.cache_data(ttl=30)
def load_network_rings() -> list[dict[str, Any]]:
    return api_get("/network/rings")


def render_return_queue() -> None:
    st.title("Return queue")
    st.caption("Ranked cases from the FastAPI scoring service.")
    queue = load_queue()
    if not queue:
        st.info("No return cases are available.")
        return

    total_cases = len(queue)
    high_risk_count = sum(row["risk_band"] == "High" for row in queue)
    high_risk_pct = high_risk_count / total_cases * 100
    total_estimated_loss = sum(
        float(
            row.get(
                "estimated_loss_if_approved",
                row["risk_score"] * row["refund_amount"],
            )
        )
        for row in queue
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric("Total cases", f"{total_cases:,}")
    summary_columns[1].metric("High risk", f"{high_risk_pct:.1f}%")
    summary_columns[2].metric(
        "Estimated loss if approved",
        f"{total_estimated_loss:,.2f}",
    )

    columns = [
        "return_id",
        "product",
        "customer",
        "risk_score",
        "risk_band",
        "reason",
        "refund_amount",
        "recommended_action",
    ]
    queue_frame = pd.DataFrame(queue)[columns]
    styled_queue_frame = queue_frame.style.map(
        lambda value: RISK_BAND_STYLES.get(str(value), ""),
        subset=["risk_band"],
    )
    st.dataframe(
        styled_queue_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "risk_score": st.column_config.NumberColumn("risk_score", format="%.3f"),
            "refund_amount": st.column_config.NumberColumn("refund_amount", format="%.2f"),
        },
    )

    selected_return_id = st.selectbox(
        "Select a return to inspect",
        options=[row["return_id"] for row in queue],
    )
    st.session_state["selected_return_id"] = selected_return_id
    st.info("Switch to **Case detail** to inspect evidence and submit a manual decision.")


def render_case_detail() -> None:
    st.title("Case detail")
    queue = load_queue()
    if not queue:
        st.info("No return cases are available.")
        return

    return_ids = [row["return_id"] for row in queue]
    default_id = st.session_state.get("selected_return_id", return_ids[0])
    default_index = return_ids.index(default_id) if default_id in return_ids else 0
    selected_return_id = st.selectbox(
        "Return case",
        options=return_ids,
        index=default_index,
    )
    st.session_state["selected_return_id"] = selected_return_id
    detail = api_get(f"/case/{selected_return_id}")
    score = detail["score"]
    order = detail["order"]
    return_record = detail["return"]

    st.subheader(f"{selected_return_id} · {order['product_id']}")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Risk score", f"{score['risk_score']:.3f}")
    metric_columns[1].metric("Risk band", score["risk_band"])
    metric_columns[2].metric("Recommended action", score["recommended_action"])
    metric_columns[3].metric(
        "Estimated loss",
        f"{score['estimated_loss_if_approved']:.2f}",
    )

    st.subheader("Order timeline")
    st.dataframe(pd.DataFrame(detail["timeline"]), hide_index=True, width="stretch")

    st.subheader("Evidence")
    evidence = detail["evidence"]
    evidence_frame = pd.DataFrame(
        [
            {"evidence": "Expected weight (g)", "value": f"{evidence['expected_weight_g']:.1f}"},
            {"evidence": "Received weight (g)", "value": f"{evidence['received_weight_g']:.1f}"},
            {"evidence": "Weight difference (g)", "value": f"{evidence['weight_difference_g']:.1f}"},
            {"evidence": "Weight within threshold", "value": str(evidence["weight_match"])},
            {"evidence": "Serial matches shipment", "value": str(evidence["serial_match"])},
            {"evidence": "Prior return count", "value": str(evidence["prior_return_count"])},
            {"evidence": "Order reason", "value": str(return_record["reason"])},
            {"evidence": "Inspection outcome", "value": str(return_record["inspection_outcome"])},
        ]
    )
    st.dataframe(evidence_frame, hide_index=True, width="stretch")

    if detail["history"]:
        st.write("Prior return history")
        st.dataframe(pd.DataFrame(detail["history"]), hide_index=True, width="stretch")
    else:
        st.info("No prior returns found for this customer.")

    st.subheader("SHAP top reasons")
    for reason in score["top_reasons"]:
        st.write(f"- {reason}")

    st.subheader("Manual override")
    st.caption("The selected decision is written to SQLite as reviewer feedback.")
    button_columns = st.columns(3)
    decisions = [
        ("Approve", "approve"),
        ("Verify", "verify"),
        ("Manual review", "manual_review"),
    ]
    for column, (label, decision) in zip(button_columns, decisions):
        if column.button(label, key=f"feedback-{selected_return_id}-{decision}"):
            feedback = api_post(
                f"/feedback/{selected_return_id}",
                json={"decision": decision},
            )
            st.success(f"Saved reviewer decision: {feedback['reviewer_decision']}")
            load_queue.clear()

    if detail["latest_feedback"]:
        st.caption(
            "Latest reviewer decision: "
            f"{detail['latest_feedback']['reviewer_decision']} "
            f"at {detail['latest_feedback']['reviewed_at']}"
        )


def render_model_performance() -> None:
    st.title("Model performance")
    performance = load_performance()
    overall = performance["overall"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Precision", f"{overall['precision']:.3f}")
    metric_columns[1].metric("Recall", f"{overall['recall']:.3f}")
    metric_columns[2].metric("F1", f"{overall['f1']:.3f}")
    metric_columns[3].metric("PR-AUC", f"{overall['pr_auc']:.3f}")
    metric_columns_2 = st.columns(3)
    metric_columns_2[0].metric("False positive rate", f"{overall['false_positive_rate']:.3f}")
    metric_columns_2[1].metric(
        f"Precision @ top {performance['top_capacity_metrics']['capacity_pct']}%",
        f"{performance['top_capacity_metrics']['precision']:.3f}",
    )
    metric_columns_2[2].metric(
        f"Recall @ top {performance['top_capacity_metrics']['capacity_pct']}%",
        f"{performance['top_capacity_metrics']['recall']:.3f}",
    )

    st.subheader("Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            performance["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        ),
        width="stretch",
    )

    st.subheader("Calibration")
    calibration = pd.DataFrame(performance["calibration"])
    if calibration.empty:
        st.info("Calibration data is not available.")
    else:
        st.line_chart(
            calibration.set_index("predicted_probability"),
            y="observed_rate",
            x_label="Predicted probability",
            y_label="Observed abuse rate",
        )

    st.subheader("Performance by risk band")
    st.dataframe(
        pd.DataFrame.from_dict(performance["by_risk_band"], orient="index"),
        width="stretch",
    )
    st.subheader("Performance by category")
    st.dataframe(
        pd.DataFrame.from_dict(performance["by_category"], orient="index"),
        width="stretch",
    )
    st.subheader("Performance by payment type")
    st.dataframe(
        pd.DataFrame.from_dict(performance["by_payment_type"], orient="index"),
        width="stretch",
    )


def render_financial_impact() -> None:
    st.title("Financial impact")
    financial = load_financial()
    baseline = financial["baseline_approve_all_loss"]
    rule_loss = financial["rule_based_policy_loss"]
    model_loss = financial["model_policy_loss"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Approve-all loss", f"{baseline:,.2f}")
    metric_columns[1].metric("Rule-policy loss", f"{rule_loss:,.2f}")
    metric_columns[2].metric("Model-policy loss", f"{model_loss:,.2f}")

    comparison = pd.DataFrame(
        {
            "policy": ["Approve all", "Rule-based", "Model-based"],
            "estimated_loss": [baseline, rule_loss, model_loss],
        }
    ).set_index("policy")
    st.bar_chart(comparison)

    st.subheader("Savings vs approve-all")
    st.dataframe(
        pd.DataFrame(
            [
                {"policy": "Rule-based", "savings": financial["savings_vs_approve_all"]["rule_based"]},
                {"policy": "Model-based", "savings": financial["savings_vs_approve_all"]["model"]},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Policy assumptions")
    st.json(financial["assumptions"])
    st.subheader("Action counts")
    st.json(financial["policy_counts"])


def render_abuse_network() -> None:
    st.title("Abuse network")
    st.caption(
        "Investigative view of customers linked by shared address, device, or payment details."
    )
    rings = [
        ring
        for ring in load_network_rings()
        if int(ring.get("ring_size", 0)) >= 2
    ]
    rings.sort(key=lambda ring: int(ring["ring_size"]), reverse=True)
    if not rings:
        st.info("No linked-account networks with two or more customers were detected.")
        return

    ring_table = pd.DataFrame(
        [
            {
                "ring_id": ring["ring_id"],
                "ring_size": ring["ring_size"],
                "member_count": len(ring["customer_ids"]),
                "shared_attributes": ", ".join(
                    NETWORK_ATTRIBUTE_LABELS.get(attribute, attribute)
                    for attribute in ring["shared_attributes"]
                ),
            }
            for ring in rings
        ]
    )
    st.dataframe(ring_table, hide_index=True, width="stretch")

    ring_by_label = {
        f"Ring {ring['ring_id']} · {ring['ring_size']} customers": ring
        for ring in rings
    }
    selection_columns = st.columns(2)
    selected_ring_label = selection_columns[0].selectbox(
        "Select a ring",
        options=list(ring_by_label),
    )
    direct_customer_id = selection_columns[1].text_input(
        "Or enter a customer ID directly",
        placeholder="e.g. CUST-000088",
    ).strip()
    selected_ring = ring_by_label[selected_ring_label]
    customer_id = direct_customer_id or selected_ring["customer_ids"][0]

    network = api_get(f"/network/{quote(customer_id, safe='')}")
    graph = nx.Graph()
    graph.add_nodes_from(network["linked_customers"] + [network["customer_id"]])
    for edge in network.get("edges", []):
        graph.add_edge(
            edge["source"],
            edge["target"],
            shared_via=edge.get("shared_via", []),
        )

    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        st.info("This customer does not have a connected network to visualize.")
        return

    history_rows = [network.get("customer_history", {})] + network.get(
        "linked_customer_history",
        [],
    )
    history_by_customer = {
        str(row["customer_id"]): row
        for row in history_rows
        if row.get("customer_id") is not None
    }
    abuse_customers = {
        customer
        for customer, row in history_by_customer.items()
        if any(int(label) == 1 for label in row.get("confirmed_abuse_labels", []))
    }

    positions = nx.spring_layout(graph, seed=42)
    figure, axis = plt.subplots(figsize=(11, 7))
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edge_color="#9aa5b1",
        width=1.8,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_color=[
            "#c95c5c" if str(node) in abuse_customers else "#5b8db8"
            for node in graph.nodes
        ],
        node_size=1800,
        edgecolors="#263238",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        ax=axis,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )
    edge_labels = {
        (edge["source"], edge["target"]): ", ".join(
            NETWORK_ATTRIBUTE_LABELS.get(attribute, attribute)
            for attribute in edge.get("shared_via", [])
        )
        for edge in network.get("edges", [])
    }
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        ax=axis,
        font_size=8,
        label_pos=0.5,
        bbox={"alpha": 0.85, "color": "white", "pad": 0.2},
    )
    axis.legend(
        handles=[
            Patch(facecolor="#c95c5c", edgecolor="#263238", label="Abuse history"),
            Patch(facecolor="#5b8db8", edgecolor="#263238", label="No abuse history"),
        ],
        loc="best",
    )
    axis.axis("off")
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    member_rows = []
    for member_id in sorted(graph.nodes, key=str):
        history = history_by_customer.get(str(member_id), {})
        labels = [int(label) for label in history.get("confirmed_abuse_labels", [])]
        member_rows.append(
            {
                "customer_id": member_id,
                "return_count": history.get("return_count", 0),
                "abuse_history": "Yes" if any(labels) else "No",
                "confirmed_abuse_labels": labels,
            }
        )
    st.subheader("Ring member history")
    st.dataframe(pd.DataFrame(member_rows), hide_index=True, width="stretch")


st.sidebar.title("ReturnShield AI")
view = st.sidebar.radio(
    "View",
    [
        "Return queue",
        "Case detail",
        "Model performance",
        "Financial impact",
        "Abuse network",
    ],
)

if view == "Return queue":
    render_return_queue()
elif view == "Case detail":
    render_case_detail()
elif view == "Model performance":
    render_model_performance()
elif view == "Financial impact":
    render_financial_impact()
else:
    render_abuse_network()