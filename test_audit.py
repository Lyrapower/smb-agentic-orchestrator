import json
import tempfile
import unittest
from pathlib import Path

from audit import get_latest_audit_entries


class AuditLogTests(unittest.TestCase):
    def test_get_latest_entries_keeps_only_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit_log.jsonl"
            with audit_path.open("w", encoding="utf-8") as handle:
                for index in range(100):
                    handle.write(json.dumps({"index": index}) + "\n")

            entries = get_latest_audit_entries(limit=5, path=audit_path)

        self.assertEqual([entry["index"] for entry in entries], [95, 96, 97, 98, 99])

    def test_get_latest_entries_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit_log.jsonl"
            audit_path.write_text(
                '{"index": 1}\nnot-json\n{"index": 2}\n',
                encoding="utf-8",
            )

            entries = get_latest_audit_entries(limit=10, path=audit_path)

        self.assertEqual([entry["index"] for entry in entries], [1, 2])

    def test_get_latest_entries_returns_empty_for_non_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit_log.jsonl"
            audit_path.write_text('{"index": 1}\n', encoding="utf-8")

            entries = get_latest_audit_entries(limit=0, path=audit_path)

        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
