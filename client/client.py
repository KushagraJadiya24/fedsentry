"""Flower client and local LoRA training functions.

Day-0 scaffold only.

Ownership:
- Kushagra: load_client_data(), train(), evaluate()
- Teammate B: FLClient

Locked CLI arguments:
    --client_id
    --server_address
"""

import argparse
from pathlib import Path
from typing import Any

import flwr as fl


DEFAULT_PARTITIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "partitions"


def load_client_data(
    client_id: int,
    partitions_dir: str = "../data/partitions",
):
    """Load this client's JSONL file and return a training-ready dataset."""
    raise NotImplementedError("Day-0 scaffold: client data loading not implemented yet.")


def train(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    epochs: int = 1,
):
    """Run local LoRA fine-tuning on this client's data."""
    raise NotImplementedError("Day-0 scaffold: local training not implemented yet.")


def evaluate(
    model: Any,
    tokenizer: Any,
    dataset: Any,
):
    """Return evaluation loss and an appropriate quality metric."""
    raise NotImplementedError("Day-0 scaffold: evaluation not implemented yet.")


class FLClient(fl.client.NumPyClient):
    """Day-0 Flower client scaffold.

    Teammate B owns this class. It will connect the locked model_utils
    and local training functions to Flower's NumPyClient interface.
    """

    def __init__(self, client_id: int):
        self.client_id = client_id
        raise NotImplementedError(
            "Day-0 scaffold: FLClient integration not implemented yet."
        )


def parse_args() -> argparse.Namespace:
    """Parse the locked client CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument(
        "--server_address",
        default="localhost:8080",
    )
    return parser.parse_args()


def main() -> None:
    """Start the Flower client."""
    args = parse_args()
    raise NotImplementedError(
        f"Day-0 scaffold: client {args.client_id} is not wired to "
        f"{args.server_address} yet."
    )


if __name__ == "__main__":
    main()
