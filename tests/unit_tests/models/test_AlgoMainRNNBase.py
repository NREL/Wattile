import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from wattile.models.AlgoMainRNNBase import AlgoMainRNNBase

TESTS_PATH = Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"


@pytest.fixture
def configs():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    return configs


@pytest.fixture
def algo_main_rnn_base(configs):
    return AlgoMainRNNBase(configs)


def test_get_input_window_for_output_time(algo_main_rnn_base):
    assert algo_main_rnn_base.get_input_window_for_output_time(datetime().now()) == (
        datetime.now(),
        datetime.now(),
    )


def test_get_pediction_vector_for_time(algo_main_rnn_base):
    assert algo_main_rnn_base.get_pediction_vector_for_time(datetime.now()) == [
        timedelta(hours=0)
    ]
