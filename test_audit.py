import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audit


class CountingPath:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.bytes_read = 0

    def exists(self) -> bool:
        return self.path.exists()

    def open(self, *args, **kwargs):
        return CountingFile(self.path.open(*args, **kwargs), self)


class CountingFile:
    def __init__(self, handle, counting_path: CountingPath) -> None:
        self.handle = handle
        self.counting_path = counting_path

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def seek(self, *args, **kwargs):
        return self.handle.seek(*args, **kwargs)

    def tell(self):
        return self.handle.tell()

    def read(self, *args, **kwargs):
        data = self.handle.read(*args, **kwargs)
        self.counting_path.bytes_read += len(data)
        return data


class AuditLogTests(unittest.TestCase):
    def test_get_latest_entries_keeps_only_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit_log.jsonl"
            with audit_path.open("w", encoding="utf-8") as handle:
                for index in range(100):
                    handle.write(json.dumps({"index": index}) + "\n")

            entries = audit.get_latest_audit_entries(limit=5, path=audit_path)

        self.assertEqual([entry["index"] for entry in entries], [95, 96, 97, 98, 99])

    def test_get_latest_entries_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit_log.jsonl"
            audit_path.write_text(
                '{"index": 1}\nnot-json\n{"index": 2}\n',
                encoding="utf-8",
            )

            entries = audit.get_latest_audit_entries(limit=10, path=audit_path)

        self.assertEqual([entry["index"] for entry in entries], [1, 2])

    def test_get_latest_entries_returns_empty_for_non_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit_log.jsonl"
            audit_path.write_text('{"index": 1}\n', encoding="utf-8")

            entries = audit.get_latest_audit_entries(limit=0, path=audit_path)

        self.assertEqual(entries, [])

    def test_get_latest_entries_reads_only_needed_tail_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit_log.jsonl"
            with audit_path.open("w", encoding="utf-8") as handle:
                for index in range(1000):
                    handle.write(json.dumps({"index": index}) + "\n")

            counting_path = CountingPath(audit_path)
            with mock.patch.object(audit, "AUDIT_READ_CHUNK_SIZE", 128):
                entries = audit.get_latest_audit_entries(limit=5, path=counting_path)

            self.assertEqual([entry["index"] for entry in entries], [995, 996, 997, 998, 999])
            self.assertLess(counting_path.bytes_read, audit_path.stat().st_size)


if __name__ == "__main__":
    unittest.main()
