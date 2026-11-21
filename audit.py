"""Append-only JSONL audit logger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_LOG_PATH = Path("data/audit_log.jsonl")
AUDIT_READ_CHUNK_SIZE = 64 * 1024


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
    if limit <= 0:
        return []

    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    pending_prefix = b""

    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()

        while position > 0 and len(entries) < limit:
            read_size = min(AUDIT_READ_CHUNK_SIZE, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)

            lines = (chunk + pending_prefix).split(b"\n")
            pending_prefix = lines[0]

            for raw_line in reversed(lines[1:]):
                if _append_audit_line(entries, raw_line, limit):
                    break

        if len(entries) < limit:
            _append_audit_line(entries, pending_prefix, limit)

    entries.reverse()
    return entries


def _append_audit_line(entries: list[dict[str, Any]], raw_line: bytes, limit: int) -> bool:
    line = raw_line.strip()
    if not line:
        return False

    try:
        entries.append(json.loads(line))
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Keep logger append-only and resilient to malformed lines.
        return False

    return len(entries) >= limit
