import json
import pathlib

import pytest

import wattile.entry_point as epb
from wattile.buildings_processing import (
    correct_predictor_columns,
    correct_timestamps,
    resample_or_rolling_stats,
)
from wattile.data_reading import read_dataset_from_file
from wattile.models.charlie_model import main as charlie_model
from wattile.time_processing import add_processed_time_columns

TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"
TESTS_DATA_PATH = TESTS_PATH / "data" / "Synthetic Site"


@pytest.fixture
def config_for_tests():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    configs["data_dir"] = str(TESTS_DATA_PATH)
    configs["data_config"] = "Synthetic Site Config.json"

    return configs


# test model training config patches
ARCH_VERSION_ALFA_RNN_CONFIG_PATCH = {
    "arch_version": "alfa",
    "arch_type_variant": "vanilla",
}
ARCH_VERSION_ALFA_LSTM_CONFIG_PATCH = {
    "arch_version": "alfa",
    "arch_type_variant": "lstm",
}
ARCH_VERSION_ALFA_STANDARD_TRANSFORMATION_CONFIG_PATCH = {
    "arch_version": "alfa",
    "transformation_method": "standard",
}

ARCH_VERSION_BRAVO_RNN_CONFIG_PATCH = {
    "arch_version": "bravo",
    "arch_type_variant": "vanilla",
}
ARCH_VERSION_BRAVO_LSTM_CONFIG_PATCH = {
    "arch_version": "bravo",
    "arch_type_variant": "lstm",
}
ARCH_VERSION_BRAVO_STANDARD_TRANSFORMATION_CONFIG_PATCH = {
    "arch_version": "bravo",
    "transformation_method": "standard",
}


@pytest.mark.parametrize(
    "config_patch",
    [
        ARCH_VERSION_ALFA_RNN_CONFIG_PATCH,
        ARCH_VERSION_ALFA_LSTM_CONFIG_PATCH,
        ARCH_VERSION_ALFA_STANDARD_TRANSFORMATION_CONFIG_PATCH,
        ARCH_VERSION_BRAVO_RNN_CONFIG_PATCH,
        ARCH_VERSION_BRAVO_LSTM_CONFIG_PATCH,
        ARCH_VERSION_BRAVO_STANDARD_TRANSFORMATION_CONFIG_PATCH,
    ],
)
def test_model_trains(config_for_tests, tmpdir, config_patch):
    """
    Run training and verify results are made.
    """
    # patch configs and create temporary, unquie output file
    config_for_tests.update(config_patch)
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)

    # train model
    epb.main(config_for_tests)

    # check result file were created
    assert (exp_dir / "output.out").exists()
    assert (exp_dir / "torch_model").exists()
    assert (exp_dir / "train_stats.json").exists()


def test_charlie_model_runs(config_for_tests, tmpdir):
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    exp_dir.mkdir()
    config_for_tests["exp_dir"] = str(exp_dir)

    data = read_dataset_from_file(config_for_tests)
    data = correct_predictor_columns(config_for_tests, data)
    data = correct_timestamps(config_for_tests, data)
    data = add_processed_time_columns(data, config_for_tests)
    data = resample_or_rolling_stats(data, config_for_tests)

    charlie_model(data=data, configs=config_for_tests)

    # check result file were created
    assert (exp_dir / "actual.csv").exists()
    assert (exp_dir / f"q{config_for_tests['qs']}.csv").exists()
