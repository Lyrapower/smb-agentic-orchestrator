"""Append-only JSONL audit logger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_LOG_PATH = Path("data/audit_log.jsonl")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_audit_entry(entry: dict[str, Any], path: Path = AUDIT_LOG_PATH) -> dict[str, Any]:
    """Append an entry to a JSONL audit log and return the saved entry."""
    _ensure_parent(path)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return payload


def get_latest_audit_entries(limit: int = 50, path: Path = AUDIT_LOG_PATH) -> list[dict[str, Any]]:
    """Read latest entries from JSONL audit log."""
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # Keep logger append-only and resilient to malformed lines.
                continue

    if limit <= 0:
        return []
    return entries[-limit:]
