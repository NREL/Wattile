import json
import pathlib
import shutil

import pandas as pd
import pytest
import xarray as xr

from wattile.data_processing import prep_for_rnn
from wattile.data_reading import read_dataset_from_file
from wattile.models import AlfaModel, BravoModel

TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"
TEST_DATA_PATH = TESTS_PATH / "data" / "Synthetic Site"
TEST_DATA_CONFIG_PATH = TEST_DATA_PATH / "Synthetic Site Config.json"


@pytest.fixture
def config_for_tests():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    return configs


def popluate_test_data_dir_with_prediction_data(data_dir):
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    data_config = json.loads(TEST_DATA_CONFIG_PATH.read_text())
    predictor_files = [
        TEST_DATA_PATH / file_["filename"]
        for file_ in data_config["files"]
        if file_["contentType"] == "predictors"
    ]

    shutil.copy(TEST_DATA_CONFIG_PATH, data_dir)
    for pf in predictor_files:
        shutil.copy(pf, data_dir)


ALFA_EXP_DIR = TESTS_FIXTURES_PATH / "alfa_exp_dir"
ALFA_CONFIG_PATCH = {"arch_version": "alfa"}

BRAVO_EXP_DIR = TESTS_FIXTURES_PATH / "bravo_exp_dir"
BRAVO_CONFIG_PATCH = {"arch_version": "bravo"}


def test_prediction_alfa(config_for_tests, tmpdir):
    # SETUP
    # don't run training
    config_for_tests["learning_algorithm"]["arch_version"] = "alfa"
    config_for_tests["learning_algorithm"]["use_case"] = "prediction"

    # use a temp result dir
    exp_dir = tmpdir / "train_results"
    config_for_tests["data_output"]["exp_dir"] = str(exp_dir)
    shutil.copytree(TESTS_FIXTURES_PATH / "alfa_exp_dir", exp_dir)

    # create a temp data dir
    config_for_tests["data_input"]["data_dir"] = str(tmpdir / "data")
    config_for_tests["data_input"]["data_config"] = "Synthetic Site Config.json"
    data_dir = tmpdir / "data"
    popluate_test_data_dir_with_prediction_data(data_dir)

    # ACTION
    data = read_dataset_from_file(config_for_tests)
    _, val_df = prep_for_rnn(config_for_tests, data)
    model = AlfaModel(config_for_tests)
    results = model.predict(val_df)

    # ASSERTION
    window_width_futurecast = pd.Timedelta(
        config_for_tests["data_processing"]["input_output_window"][
            "window_width_futurecast"
        ]
    )
    quantiles = config_for_tests["learning_algorithm"]["quantiles"]

    # assert results is the right shape
    assert isinstance(results, xr.DataArray)
    assert results.shape == (val_df.shape[0], len(quantiles), 1)
    assert results.dims == ("timestamp", "quantile", "horizon")

    # assert results is indexed correctly
    timestamp, quantile, horizon = results.indexes.values()
    assert timestamp.to_list() == val_df.index.to_list()
    assert quantile.to_list() == quantiles
    assert horizon.to_list() == [window_width_futurecast]

    # assert output file was made
    assert (exp_dir / "output.out").exists()


def test_prediction_bravo(config_for_tests, tmpdir):
    # don't run training
    config_for_tests["learning_algorithm"]["arch_version"] = "bravo"
    config_for_tests["learning_algorithm"]["use_case"] = "prediction"

    # use a temp result dir
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["data_output"]["exp_dir"] = str(exp_dir)
    shutil.copytree(TESTS_FIXTURES_PATH / "bravo_exp_dir", exp_dir)

    # create a temp data dir
    config_for_tests["data_input"]["data_dir"] = str(tmpdir / "data")
    config_for_tests["data_input"]["data_config"] = "Synthetic Site Config.json"
    data_dir = tmpdir / "data"
    popluate_test_data_dir_with_prediction_data(data_dir)

    # ACTION
    data = read_dataset_from_file(config_for_tests)
    _, val_df = prep_for_rnn(config_for_tests, data)
    model = BravoModel(config_for_tests)
    results = model.predict(val_df)

    # ASSERTION
    cdp = config_for_tests["data_processing"]
    window_width_futurecast = cdp["input_output_window"]["window_width_futurecast"]
    window_width_target = pd.Timedelta(
        cdp["input_output_window"]["window_width_target"]
    )
    bin_interval = pd.Timedelta(cdp["resample"]["bin_interval"])
    quantiles = config_for_tests["learning_algorithm"]["quantiles"]

    num_lagged_targets = int(window_width_target / bin_interval) + 1

    # assert results is the right shape
    assert isinstance(results, xr.DataArray)
    assert results.shape == (val_df.shape[0], len(quantiles), num_lagged_targets)
    assert results.dims == ("timestamp", "quantile", "horizon")

    # assert results is indexed correctly
    timestamp, quantile, horizon = results.indexes.values()
    assert timestamp.to_list() == val_df.index.to_list()
    assert quantile.to_list() == quantiles
    assert horizon.to_list() == [
        window_width_futurecast + bin_interval * i for i in range(num_lagged_targets)
    ]

    assert (exp_dir / "output.out").exists()
