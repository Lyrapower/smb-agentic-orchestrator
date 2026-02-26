# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

**smb-agentic-orchestrator** is a FastAPI service for appointment management via chat. It performs rule-based intent classification (schedule / reschedule / cancel) and logs actions to an append-only JSONL audit file at `data/audit_log.jsonl`.

### Running the dev server

```bash
source .venv/bin/activate
python3 -m uvicorn main:app --reload --port 8000
```

Swagger UI is available at `http://127.0.0.1:8000/docs`.

### API endpoints

See `README.md` for full endpoint documentation (`POST /chat`, `POST /execute`, `GET /audit`).

### Linting

```bash
source .venv/bin/activate
ruff check .
```

`pyright` reports one pre-existing type-narrowing warning in `main.py:42` (`route_intent` returns `str` instead of `Intent` literal). This is a known issue in the codebase and does not affect runtime behavior.

### Gotchas

- The `data/` directory is auto-created at runtime by `audit.py`; do not manually create `audit_log.jsonl`.
- The project has no automated test suite yet. Verify changes via `curl` or Swagger UI.
- `python3.12-venv` must be installed as a system package (`sudo apt-get install -y python3.12-venv`) before creating the virtualenv; the update script handles virtualenv creation and dependency install.
