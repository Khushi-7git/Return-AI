"""SQLite persistence for orders, returns, scores, and reviewer feedback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATABASE_PATH = DATA_DIR / "returnshield.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


class Base(DeclarativeBase):
    """Base class for SQLite ORM models."""


class OrderRecord(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    timestamp: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    price: Mapped[float] = mapped_column(Float)
    payment_type: Mapped[str] = mapped_column(String(32))
    pincode: Mapped[str] = mapped_column(String(16))
    hashed_address: Mapped[str] = mapped_column(String(64), index=True)
    hashed_device: Mapped[str] = mapped_column(String(64), index=True)
    hashed_payment: Mapped[str] = mapped_column(String(64))
    expected_weight_g: Mapped[float] = mapped_column(Float)
    shipped_serial: Mapped[str] = mapped_column(String(128))
    delivery_status: Mapped[str] = mapped_column(String(32))


class ReturnRecord(Base):
    __tablename__ = "returns"

    return_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    timestamp: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(64))
    comment: Mapped[str] = mapped_column(Text)
    days_after_delivery: Mapped[int] = mapped_column(Integer)
    pickup_status: Mapped[str] = mapped_column(String(32))
    refund_amount: Mapped[float] = mapped_column(Float)
    received_weight_g: Mapped[float] = mapped_column(Float)
    received_serial: Mapped[str] = mapped_column(String(128))
    packaging_condition: Mapped[str] = mapped_column(String(32))
    inspection_outcome: Mapped[str] = mapped_column(String(64))
    confirmed_abuse_label: Mapped[int] = mapped_column(Integer)


class ScoreRecord(Base):
    __tablename__ = "scores"

    return_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16))
    recommended_action: Mapped[str] = mapped_column(String(32))
    estimated_loss_if_approved: Mapped[float] = mapped_column(Float)
    top_reasons_json: Mapped[str] = mapped_column(Text)
    recommended_verification: Mapped[str] = mapped_column(Text)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewerFeedback(Base):
    __tablename__ = "reviewer_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    return_id: Mapped[str] = mapped_column(String(32), index=True)
    reviewer_decision: Mapped[str] = mapped_column(String(32))
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


DATA_COLUMNS = {
    "orders": [
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
    ],
    "returns": [
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
    ],
}

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def initialize_database() -> None:
    """Create tables and load the generated CSVs once if the DB is empty."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        if session.scalar(select(OrderRecord.order_id).limit(1)) is None:
            orders_path = DATA_DIR / "orders.csv"
            returns_path = DATA_DIR / "returns.csv"
            if not orders_path.exists() or not returns_path.exists():
                raise FileNotFoundError(
                    "Generated CSVs are missing. Run generate_dataset() before starting the API."
                )
            orders = pd.read_csv(orders_path).where(pd.notna, None)
            returns = pd.read_csv(returns_path).where(pd.notna, None)
            session.add_all(
                [OrderRecord(**row) for row in orders[DATA_COLUMNS["orders"]].to_dict("records")]
            )
            session.add_all(
                [ReturnRecord(**row) for row in returns[DATA_COLUMNS["returns"]].to_dict("records")]
            )
            session.commit()


def _model_to_dict(record: Any) -> dict[str, Any]:
    return {
        column.name: getattr(record, column.name)
        for column in record.__table__.columns
    }


def get_order(order_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        record = session.get(OrderRecord, order_id)
        return _model_to_dict(record) if record else None


def get_return(return_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        record = session.get(ReturnRecord, return_id)
        return _model_to_dict(record) if record else None


def get_order_for_return(return_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        record = session.scalar(
            select(OrderRecord)
            .join(ReturnRecord, ReturnRecord.order_id == OrderRecord.order_id)
            .where(ReturnRecord.return_id == return_id)
        )
        return _model_to_dict(record) if record else None


def get_customer_history(customer_id: str, exclude_return_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        records = session.scalars(
            select(ReturnRecord)
            .join(OrderRecord, ReturnRecord.order_id == OrderRecord.order_id)
            .where(
                OrderRecord.customer_id == customer_id,
                ReturnRecord.return_id != exclude_return_id,
            )
            .order_by(ReturnRecord.timestamp, ReturnRecord.return_id)
        ).all()
        return [_model_to_dict(record) for record in records]


def list_cases() -> list[dict[str, Any]]:
    """Return return/order rows used by the API queue and analytics views."""
    with SessionLocal() as session:
        rows = session.execute(
            select(ReturnRecord, OrderRecord).join(
                OrderRecord, ReturnRecord.order_id == OrderRecord.order_id
            )
        ).all()
        cases: list[dict[str, Any]] = []
        for return_record, order_record in rows:
            case = _model_to_dict(return_record)
            case.update(
                {
                    "customer_id": order_record.customer_id,
                    "product_id": order_record.product_id,
                    "category": order_record.category,
                    "price": order_record.price,
                    "hashed_address": order_record.hashed_address,
                    "hashed_device": order_record.hashed_device,
                    "expected_weight_g": order_record.expected_weight_g,
                    "shipped_serial": order_record.shipped_serial,
                }
            )
            cases.append(case)
        return cases


def save_score(return_id: str, order_id: str, payload: dict[str, Any]) -> None:
    with SessionLocal() as session:
        record = session.get(ScoreRecord, return_id)
        if record is None:
            record = ScoreRecord(return_id=return_id, order_id=order_id)
            session.add(record)
        record.order_id = order_id
        record.risk_score = float(payload["risk_score"])
        record.risk_band = str(payload["risk_band"])
        record.recommended_action = str(payload["recommended_action"])
        record.estimated_loss_if_approved = float(payload["estimated_loss_if_approved"])
        record.top_reasons_json = json.dumps(payload["top_reasons"])
        record.recommended_verification = str(payload["recommended_verification"])
        record.scored_at = datetime.now(timezone.utc).replace(tzinfo=None)
        session.commit()


def save_feedback(return_id: str, decision: str) -> dict[str, Any]:
    with SessionLocal() as session:
        record = ReviewerFeedback(
            return_id=return_id,
            reviewer_decision=decision,
            reviewed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(record)
        session.commit()
        return _model_to_dict(record)


def get_latest_feedback(return_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        record = session.scalar(
            select(ReviewerFeedback)
            .where(ReviewerFeedback.return_id == return_id)
            .order_by(ReviewerFeedback.reviewed_at.desc(), ReviewerFeedback.id.desc())
        )
        return _model_to_dict(record) if record else None