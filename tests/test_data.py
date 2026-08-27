"""Contract tests for the locked client data format."""

import json
from pathlib import Path


PARTITIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "partitions"


def test_partition_files_exist():
    """The three locked partition paths should exist after data preparation."""
    for client_id in range(3):
        assert (PARTITIONS_DIR / f"client_{client_id}.jsonl").exists()


def test_partition_schema():
    """Every JSONL record must contain instruction and response."""
    for client_id in range(3):
        path = PARTITIONS_DIR / f"client_{client_id}.jsonl"
        if not path.exists():
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            assert set(("instruction", "response")).issubset(record)
            assert isinstance(record["instruction"], str)
            assert isinstance(record["response"], str)
