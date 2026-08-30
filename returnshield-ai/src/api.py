"""Minimal FastAPI application for environment verification."""

from fastapi import FastAPI

app = FastAPI(title="ReturnShield AI")


@app.get("/")
def hello_world() -> dict[str, str]:
    """Return a small response proving the API is running."""
    return {"message": "Hello from ReturnShield AI"}