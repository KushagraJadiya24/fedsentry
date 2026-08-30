import argparse

import flwr as fl
import numpy as np


class DummyClient(fl.client.NumPyClient):
    """Simple dummy Flower client for testing the server."""

    def __init__(self, client_id: int):
        self.client_id = client_id

        # Simple dummy model parameters
        self.parameters = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ]

    def get_parameters(self, config):
        return self.parameters

    def fit(self, parameters, config):
        print(f"Client {self.client_id}: training")

        # Pretend to train by slightly changing parameters
        self.parameters = [
            parameter + 0.1 * self.client_id
            for parameter in parameters
        ]

        num_examples = 10 * self.client_id

        return (
            self.parameters,
            num_examples,
            {
                "accuracy": 0.8 + 0.01 * self.client_id
            },
        )

    def evaluate(self, parameters, config):
        print(f"Client {self.client_id}: evaluating")

        # Dummy loss
        loss = 1.0 / self.client_id
        num_examples = 10 * self.client_id

        return (
            loss,
            num_examples,
            {
                "accuracy": 0.75 + 0.01 * self.client_id
            },
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--server_address",
        default="localhost:8080",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    client = DummyClient(args.client_id)

    print(
        f"Starting Dummy Client {args.client_id} "
        f"and connecting to {args.server_address}"
    )

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()