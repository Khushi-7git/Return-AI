---
name: Python dependency resolver
description: Python package installation behavior for projects that require SHAP.
---

The managed Python resolver may reject an otherwise valid Python 3.13 dependency set when `shap` is included because it evaluates an unsupported future-Python split. A project-local virtual environment can resolve and install the same requirements successfully.

**Why:** The resolver's supported-version matrix can be broader than the interpreter actually used by the project.

**How to apply:** When a requested Python dependency set fails only during global resolution, keep the requested requirements intact and install against the project's active `.venv` interpreter.