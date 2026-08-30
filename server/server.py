"""
FedSentry Flower Server

Federated Server + Evaluation using FedAvg.

Locked CLI arguments:
    --rounds
    --min_clients
    --address

Locked metrics schema:
    round,loss,num_clients
"""

import argparse
import csv
from pathlib import Path

import flwr as fl
from flwr.server.strategy import FedAvg


# Save metrics inside the server folder
METRICS_FILE = Path(__file__).with_name("metrics.csv")


def init_metrics_file():
    """Create the metrics CSV with the locked header."""
    with open(METRICS_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["round", "loss", "num_clients"])


def weighted_average(metrics):
    """
    Aggregate client metrics using the number of examples as weights.

    Expected input:
        [
            (num_examples, {"accuracy": value}),
            (num_examples, {"accuracy": value}),
            ...
        ]
    """
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    aggregated_metrics = {}

    # Collect all metric names
    metric_names = set()
    for _, client_metrics in metrics:
        metric_names.update(client_metrics.keys())

    # Compute weighted average for numeric metrics
    for metric_name in metric_names:
        weighted_sum = 0.0
        total_weight = 0

        for num_examples, client_metrics in metrics:
            value = client_metrics.get(metric_name)

            if isinstance(value, (int, float)):
                weighted_sum += num_examples * value
                total_weight += num_examples

        if total_weight > 0:
            aggregated_metrics[metric_name] = weighted_sum / total_weight

    return aggregated_metrics


class MetricsFedAvg(FedAvg):
    """FedAvg strategy that also logs evaluation metrics to CSV."""

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and log round metrics."""

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if loss_aggregated is not None:
            num_clients = len(results)

            with open(
                METRICS_FILE,
                mode="a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        server_round,
                        loss_aggregated,
                        num_clients,
                    ]
                )

            print(
                f"Round {server_round}: "
                f"loss={loss_aggregated:.6f}, "
                f"num_clients={num_clients}"
            )

        return loss_aggregated, metrics_aggregated


def parse_args() -> argparse.Namespace:
    """Parse the locked server CLI arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="Number of federated training rounds",
    )

    parser.add_argument(
        "--min_clients",
        type=int,
        required=True,
        help="Minimum number of clients required",
    )

    parser.add_argument(
        "--address",
        default="0.0.0.0:8080",
        help="Flower server address",
    )

    return parser.parse_args()


def main() -> None:
    """Start the Flower FedAvg server."""

    args = parse_args()

    # Create/reset metrics file
    init_metrics_file()

    # Configure FedAvg
    strategy = MetricsFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print("=" * 60)
    print("Starting FedSentry Flower Server")
    print(f"Address      : {args.address}")
    print(f"Rounds       : {args.rounds}")
    print(f"Min Clients  : {args.min_clients}")
    print("Strategy     : FedAvg")
    print("=" * 60)

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(
            num_rounds=args.rounds
        ),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()