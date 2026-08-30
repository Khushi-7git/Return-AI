"""Smoke tests for the project skeleton."""

from src.api import app, hello_world


def test_hello_world_endpoint() -> None:
    assert app.title == "ReturnShield AI"
    assert hello_world() == {"message": "Hello from ReturnShield AI"}