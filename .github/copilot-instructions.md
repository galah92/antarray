## applyTo: '\*\*'

---
Do not add dependencies to the `pyproject.toml` file unless explicitly requested. You may suggest dependencies in your responses.
---

When writing code, follow these guidelines:

- Use `uv run script.py` to execute python scripts or commands.
- Use `uv run python -m module_name` to run Python modules.
- Do not use `python` or `python3` directly; always use `uv run` to ensure the correct environment is used.

---

After editing code, make sure to run the following check commands to ensure code quality and consistency:

- `uv run ruff check`
- `uv run ruff format --diff`
- `uv run ty check`

Try to run these commands together to ensure all checks are performed in one go.