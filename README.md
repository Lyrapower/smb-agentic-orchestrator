# Chat Audit API Service

Simple FastAPI service with:

- `POST /chat` for rule-based intent detection (`schedule | reschedule | cancel`)
- `POST /execute` for simulated action execution + audit logging
- `GET /audit` for latest 50 audit entries

## Run steps (exact)

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --reload --port 8000
```

## API

### POST /chat

Request:

```json
{"text": "Please move my appointment to next week"}
```

Response:

```json
{
  "intent": "reschedule",
  "reason": "Matched 1 keyword(s) for intent 'reschedule'.",
  "text": "Please move my appointment to next week"
}
```

### POST /execute

Request:

```json
{"intent":"cancel","text":"Cancel my appointment for tomorrow"}
```

Writes an append-only JSONL audit entry to `data/audit_log.jsonl`.

### GET /audit

Returns the latest 50 audit entries as JSON.

## Docs

After running the server:

- Swagger UI: `http://127.0.0.1:8000/docs`