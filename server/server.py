"""Flower FedAvg server and metrics logging.

Day-0 scaffold only.

Locked CLI arguments:
    --rounds
    --min_clients
    --address

Locked metrics schema:
    round, loss, num_clients
"""

import argparse


METRICS_FILE = "metrics.csv"


def init_metrics_file():
    """Create the metrics CSV with the locked header."""
    raise NotImplementedError("Day-0 scaffold: metrics initialization not implemented yet.")


def weighted_average(metrics):
    """Aggregate client metrics into a weighted average."""
    raise NotImplementedError("Day-0 scaffold: metric aggregation not implemented yet.")


def parse_args() -> argparse.Namespace:
    """Parse the locked server CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--min_clients", type=int, required=True)
    parser.add_argument("--address", default="0.0.0.0:8080")
    return parser.parse_args()


def main() -> None:
    """Start the Flower FedAvg server."""
    args = parse_args()
    raise NotImplementedError(
        f"Day-0 scaffold: server not wired yet "
        f"(rounds={args.rounds}, min_clients={args.min_clients}, "
        f"address={args.address})."
    )


if __name__ == "__main__":
    main()
