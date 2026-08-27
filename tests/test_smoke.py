"""Day-0 smoke tests for repository contracts."""

import inspect

from client.client import evaluate, load_client_data, train
from server.server import METRICS_FILE, weighted_average


def test_client_interfaces_exist():
    """Locked client-side functions exist with callable signatures."""
    assert callable(load_client_data)
    assert callable(train)
    assert callable(evaluate)


def test_server_contract():
    """Locked metrics filename and aggregation function exist."""
    assert METRICS_FILE == "metrics.csv"
    assert callable(weighted_average)


def test_client_cli_names_are_present():
    """Client CLI names remain part of the scaffold."""
    source = inspect.getsource(__import__("client.client", fromlist=["main"]))
    assert "--client_id" in source
    assert "--server_address" in source


def test_server_cli_names_are_present():
    """Server CLI names remain part of the scaffold."""
    source = inspect.getsource(__import__("server.server", fromlist=["main"]))
    assert "--rounds" in source
    assert "--min_clients" in source
    assert "--address" in source
