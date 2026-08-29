"""Prepare the three locked non-IID JSONL client partitions.

Owner: Kashish( Data + Client Wiring )

Day-0 scaffold only.

Final implementation requirements:
- Generate:
    data/partitions/client_0.jsonl
    data/partitions/client_1.jsonl
    data/partitions/client_2.jsonl
- Each line must be:
    {"instruction": "...", "response": "..."}
- The three shards must be non-IID.
- Document the source dataset and split method in this module docstring.

Run from the repository root:
    python data/prepare_data.py
"""

from pathlib import Path


PARTITIONS_DIR = Path(__file__).resolve().parent / "partitions"


def main() -> None:
    """Generate the three client partitions.

    TODO: Implement dataset download/loading, non-IID split, validation,
    and JSONL writing.
    """
    raise NotImplementedError("Day-0 scaffold: data preparation not implemented yet.")


if __name__ == "__main__":
    main()
